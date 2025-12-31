"""
Fast training script using C++ backend for self-play.
"""

import sys
import os
import time
import glob
import torch
import numpy as np
from torch import nn, optim

# -------------------------------------------------
# C++ backend
# -------------------------------------------------
sys.path.append(os.path.abspath("./cpp"))
import echelon_cpp

from model import EchelonNet
from selfplay import ReplayBuffer

# Initialize C++ tables once
echelon_cpp.init()


# -------------------------------------------------
# Model → C++ bridge
# -------------------------------------------------
class FastModelWrapper:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, board_tensor):
        tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, policy_logits = self.model(tensor)
        return value.item(), policy_logits.squeeze(0).cpu().numpy()


# -------------------------------------------------
# Self-play (C++ driven)
# -------------------------------------------------
def play_game_cpp(
    model,
    num_simulations=100,
    max_moves=300,
    device="cpu",
):
    wrapper = FastModelWrapper(model, device)
    mcts = echelon_cpp.MCTS(num_simulations=num_simulations)

    board = echelon_cpp.BoardState()
    board.parse_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    positions = []
    policies = []
    sides = []
    seen_positions = set()

    for move_num in range(max_moves):
        # repetition guard
        # prefer explicit FEN getter from C++ bindings
        fen = board.to_fen() if hasattr(board, "to_fen") else str(board)
        if fen in seen_positions:
            outcome = 0.0
            break
        seen_positions.add(fen)

        legal_moves = board.generate_legal_moves()
        if not legal_moves:
            outcome = -1.0 if board.is_in_check() else 0.0
            break

        # temperature annealing
        if move_num < 20:
            mcts.set_temperature(1.0)
        elif move_num < 40:
            mcts.set_temperature(0.5)
        else:
            mcts.set_temperature(0.1)

        move_probs = mcts.search(board, wrapper)

        board_tensor = torch.from_numpy(board.tensorize()).cpu()
        policy = np.zeros(4672, dtype=np.float32)
        for idx, prob in move_probs.items():
            policy[idx] = prob

        # robust side-to-move detection (0 = white, 1 = black)
        def _get_side(b):
            for name in ("side_to_move", "side", "to_move", "turn", "white_to_move", "sideToMove"):
                attr = getattr(b, name, None)
                if attr is None:
                    continue
                try:
                    v = attr() if callable(attr) else attr
                except Exception:
                    continue

                if isinstance(v, (int,)) and v in (0, 1):
                    return int(v)
                if isinstance(v, bool):
                    return int(v)
                sv = str(v).lower()
                if sv.startswith("w"):
                    return 0
                if sv.startswith("b"):
                    return 1
            raise AttributeError("BoardState has no recognizable side-to-move accessor")

        side = _get_side(board)
        positions.append(board_tensor)
        policies.append(policy)
        sides.append(side)

        best_idx = max(move_probs, key=move_probs.get)

        moved = False
        for m in legal_moves:
            if mcts.encode_move(m) == best_idx:
                board.make_move(m)
                moved = True
                break

        if not moved:
            outcome = 0.0
            break

        # early value cutoff
        if move_num > 80:
            value_estimate, _ = wrapper.predict(board.tensorize())
            if abs(value_estimate) > 0.95:
                outcome = 1.0 if value_estimate > 0 else -1.0
                break
    else:
        outcome = 0.0

    # build training examples
    examples = []
    for pos, pol, side in zip(positions, policies, sides):
        value = outcome if side == 0 else -outcome
        examples.append({
            "position": pos,
            "policy": torch.from_numpy(pol).float().cpu(),
            "value": torch.tensor([[value]], dtype=torch.float32),
        })

    return examples


# -------------------------------------------------
# Training step
# -------------------------------------------------
def train_iteration(
    model,
    optimizer,
    replay_buffer,
    batch_size=128,
    num_batches=500,
):
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0
    max_batches = min(num_batches, len(replay_buffer) // batch_size)

    for _ in range(max_batches):
        positions, policies, values = replay_buffer.sample(batch_size)

        positions = positions.to(device)
        policies = policies.to(device)
        values = values.to(device)

        optimizer.zero_grad()
        pred_values, pred_policies = model(positions)

        value_loss = nn.MSELoss()(pred_values, values)
        policy_loss = -torch.mean(
            torch.sum(
                policies * nn.functional.log_softmax(pred_policies, dim=1),
                dim=1,
            )
        )

        loss = policy_loss + 0.5 * value_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max_batches if max_batches > 0 else 0.0


# -------------------------------------------------
# Checkpoint utils
# -------------------------------------------------
def find_latest_checkpoint():
    checkpoints = glob.glob("checkpoint_iter_*.pt") + \
                  glob.glob("checkpoints/checkpoint_iter_*.pt")

    best = None
    for ckpt in checkpoints:
        try:
            n = int(os.path.basename(ckpt).split("_")[-1].replace(".pt", ""))
            if best is None or n > best[0]:
                best = (n, ckpt)
        except ValueError:
            pass

    return best


def load_checkpoint(path, model, optimizer, scheduler, device):
    print(f"\nLoading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    replay_buffer = ReplayBuffer(max_size=50_000)

    if "replay_buffer" in checkpoint:
        for ex in checkpoint["replay_buffer"]:
            ex["position"] = ex["position"].cpu()
            ex["policy"] = ex["policy"].cpu()
            ex["value"] = ex["value"].cpu()
            replay_buffer.buffer.append(ex)

        print(f"  Restored replay buffer with {len(replay_buffer)} examples")

    iteration = checkpoint.get("iteration", 0)
    print(f"  Resumed from iteration {iteration}")

    return iteration, replay_buffer


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    replay_buffer,
    iteration,
):
    path = f"checkpoint_iter_{iteration}.pt"
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "replay_buffer": list(replay_buffer.buffer),
        },
        path,
    )
    print(f"  Saved checkpoint: {path}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print("=" * 70)
    print("ECHELON TRAINING WITH C++ BACKEND")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EchelonNet(
        in_channels=13,
        num_res_blocks=5,
        num_filters=128,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    replay_buffer = ReplayBuffer(max_size=50_000)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_iteration = 0
    latest = find_latest_checkpoint()
    if latest:
        start_iteration, replay_buffer = load_checkpoint(
            latest[1], model, optimizer, scheduler, device
        )
        start_iteration += 1
        print(f"\n✓ Resuming training from iteration {start_iteration}")
    else:
        print("\n✓ Starting fresh training")

    iterations_to_run = 10
    for iteration in range(start_iteration, start_iteration + iterations_to_run):
        print(f"\n--- Iteration {iteration} ---")

        print("[1/2] Generating self-play games...")
        t0 = time.time()
        for g in range(5):
            examples = play_game_cpp(
                model,
                num_simulations=50,
                max_moves=300,
                device=device,
            )
            replay_buffer.add_examples(examples)
            print(f"  Game {g+1}/5: {len(examples)} positions")

        print(f"  Self-play time: {time.time() - t0:.1f}s")
        print(f"  Buffer size: {len(replay_buffer)}")

        print("[2/2] Training neural network...")
        t0 = time.time()
        loss = train_iteration(
            model,
            optimizer,
            replay_buffer,
            batch_size=128,
            num_batches=500,
        )
        print(f"  Loss: {loss:.4f}")
        print(f"  Training time: {time.time() - t0:.1f}s")

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            replay_buffer,
            iteration,
        )

        print(f"  LR before step: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
