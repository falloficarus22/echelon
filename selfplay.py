import sys
import os
import time
import numpy as np
import torch
from collections import deque

# Add C++ backend to path
sys.path.append(os.path.abspath("./cpp"))
import echelon_cpp


# -----------------------------
# Model → C++ bridge
# -----------------------------
class FastModelWrapper:
    """Bridges PyTorch model with C++ MCTS"""
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, board_tensor):
        """
        Called from C++ MCTS
        board_tensor: np.ndarray [13, 8, 8]
        """
        tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, policy_logits = self.model(tensor)
        return value.item(), policy_logits.squeeze(0).cpu().numpy()


# -----------------------------
# Self-play game container
# -----------------------------
class SelfPlayGame:
    def __init__(self):
        self.positions = []
        self.policies = []
        self.sides = []

    def add_position(self, position, policy, side):
        self.positions.append(position)
        self.policies.append(policy)
        self.sides.append(side)

    def get_training_data(self, outcome):
        """
        outcome is from white's perspective:
          +1 white win
          -1 black win
           0 draw
        """
        examples = []
        for pos, pol, side in zip(self.positions, self.policies, self.sides):
            # Flip value if black was to move
            value = outcome if side == 0 else -outcome
            examples.append({
                "position": pos,
                "policy": pol,
                "value": torch.tensor([[value]], dtype=torch.float32)
            })
        return examples


# -----------------------------
# Self-play worker
# -----------------------------
class SelfPlayWorker:
    def __init__(
        self,
        model,
        num_simulations=800,
        max_game_length=512,
        device="cpu",
    ):
        self.fast_model = FastModelWrapper(model, device)
        self.mcts = echelon_cpp.MCTS(num_simulations=num_simulations)
        self.max_game_length = max_game_length

    def play_game(self, verbose=False):
        board = echelon_cpp.BoardState()
        board.parse_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )

        game = SelfPlayGame()
        seen_positions = set()

        for move_count in range(self.max_game_length):
            # --- repetition detection (cheap & effective)
            fen = board.to_fen()
            if fen in seen_positions:
                outcome = 0.0
                break
            seen_positions.add(fen)

            legal_moves = board.generate_legal_moves()

            # --- terminal: no legal moves
            if not legal_moves:
                # side to move has lost if in check
                if board.is_in_check():
                    outcome = -1.0
                else:
                    outcome = 0.0
                break

            # --- temperature annealing
            if move_count < 20:
                self.mcts.set_temperature(1.0)
            elif move_count < 40:
                self.mcts.set_temperature(0.5)
            else:
                self.mcts.set_temperature(0.1)

            # --- MCTS
            move_probs = self.mcts.search(board, self.fast_model)

            # --- store training data
            board_tensor = torch.from_numpy(board.tensorize()).cpu()
            policy = np.zeros(4672, dtype=np.float32)
            for idx, prob in move_probs.items():
                policy[idx] = prob

            # side: 0 = white, 1 = black — use robust accessor
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

            game.add_position(
                board_tensor,
                torch.from_numpy(policy).float().cpu(),
                side
            )

            # --- pick move
            best_idx = max(move_probs, key=move_probs.get)

            # --- execute move
            moved = False
            for m in legal_moves:
                if self.mcts.encode_move(m) == best_idx:
                    board.make_move(m)
                    moved = True
                    break

            if not moved:
                # should never happen
                outcome = 0.0
                break

            # --- early value-based termination
            if move_count > 80:
                value_estimate, _ = self.fast_model.predict(board.tensorize())
                if abs(value_estimate) > 0.95:
                    outcome = 1.0 if value_estimate > 0 else -1.0
                    break
        else:
            outcome = 0.0  # draw by length

        examples = game.get_training_data(outcome)

        if verbose:
            print(f"Game finished: {len(examples)} positions")
            print(f"Outcome: {outcome}")

        return examples

    def generate_games(self, num_games, verbose=False):
        all_examples = []
        for i in range(num_games):
            start = time.time()
            examples = self.play_game(verbose=(verbose and i == 0))
            all_examples.extend(examples)
            print(
                f"Game {i+1}/{num_games}: "
                f"{len(examples)} positions "
                f"({time.time() - start:.1f}s)"
            )
        return all_examples


# -----------------------------
# Replay buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, max_size=500_000):
        self.buffer = deque(maxlen=max_size)

    def add_examples(self, examples):
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        positions = torch.stack([b["position"] for b in batch])
        policies = torch.stack([b["policy"] for b in batch])
        values = torch.stack([b["value"] for b in batch])
        return positions, policies, values

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# Quick test
# -----------------------------
def test_selfplay():
    from model import EchelonNet

    model = EchelonNet(
        in_channels=13,
        num_res_blocks=5,
        num_filters=128
    )

    worker = SelfPlayWorker(
        model,
        num_simulations=50,
        max_game_length=200,
    )

    examples = worker.play_game(verbose=True)
    print(f"Generated {len(examples)} examples")


if __name__ == "__main__":
    test_selfplay()
