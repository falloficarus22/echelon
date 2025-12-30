"""
Fast training script using C++ backend for self-play.
This replaces the slow Python engine with the high-performance C++ implementation.
"""

import sys
import os
import torch
import numpy as np
from torch import nn, optim
import time

# Add C++ backend
sys.path.append(os.path.abspath("./cpp"))
import echelon_cpp

from model import EchelonNet
from selfplay import ReplayBuffer

# Initialize C++ tables once
echelon_cpp.init()

class FastModelWrapper:
    """Efficient bridge between PyTorch and C++ MCTS"""
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict(self, board_tensor):
        """Called by C++ MCTS during search"""
        tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, policy_logits = self.model(tensor)
        return value.item(), policy_logits.squeeze(0).cpu().numpy()

def play_game_cpp(model, num_simulations=100, max_moves=200, device="cpu"):
    """Play one self-play game using C++ backend"""
    wrapper = FastModelWrapper(model, device=device)
    mcts = echelon_cpp.MCTS(num_simulations=num_simulations)
    
    board = echelon_cpp.BoardState()
    board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    positions = []
    policies = []
    
    for move_num in range(max_moves):
        # Check game over
        legal_moves = board.generate_legal_moves()
        if not legal_moves:
            outcome = -1.0 if board.is_in_check() else 0.0
            break
        
        # Run MCTS search
        move_probs = mcts.search(board, wrapper)
        
        # Store training data
        tensor = torch.from_numpy(board.tensorize())
        policy = np.zeros(4672, dtype=np.float32)
        for idx, prob in move_probs.items():
            policy[idx] = prob
        
        positions.append(tensor)
        policies.append(policy)
        
        # Make best move
        best_idx = max(move_probs, key=move_probs.get)
        
        # Find corresponding move and execute it
        moved = False
        for m in legal_moves:
            if mcts.encode_move(m) == best_idx:
                board.make_move(m)
                moved = True
                break
        
        if not moved:
            print(f"Warning: Could not find move for index {best_idx}")
            break
    else:
        outcome = 0.0  # Draw by length
    
    # Convert to training examples
    examples = []
    for pos, pol in zip(positions, policies):
        examples.append({
            'position': pos,
            'policy': torch.tensor(pol),
            'value': torch.tensor([[outcome]], dtype=torch.float32)
        })
    
    return examples

def train_iteration(model, optimizer, replay_buffer, batch_size=64, num_batches=100):
    """Train on replay buffer"""
    model.train()
    device = next(model.parameters()).device
    
    total_loss = 0
    for _ in range(num_batches):
        if len(replay_buffer) < batch_size:
            break
            
        positions, policies, values = replay_buffer.sample(batch_size)
        
        # Move to device (GPU)
        positions = positions.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        optimizer.zero_grad()
        pred_values, pred_policies = model(positions)
        
        value_loss = nn.MSELoss()(pred_values, values)
        policy_loss = -torch.mean(torch.sum(policies * nn.functional.log_softmax(pred_policies, dim=1), dim=1))
        
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches if num_batches > 0 else 0

def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    import glob
    import os
    
    # 1. Look for checkpoints from this script (train_cpp.py)
    # Search in both current directory and checkpoints/ folder
    checkpoints = glob.glob("checkpoint_iter_*.pt") + glob.glob("checkpoints/checkpoint_iter_*.pt")
    
    if checkpoints:
        # Extract iteration numbers and find the latest
        iterations = []
        for ckpt in checkpoints:
            try:
                # Handle paths like "checkpoints/checkpoint_iter_10.pt" correctly
                filename = os.path.basename(ckpt)
                iter_num = int(filename.split("_")[-1].replace(".pt", ""))
                iterations.append((iter_num, ckpt))
            except ValueError:
                continue
        
        if iterations:
            iterations.sort(reverse=True)
            return iterations[0]  # (iteration_number, checkpoint_path)

    # 2. Look for checkpoints from previous python training (train.py) in checkpoints/
    if os.path.exists("checkpoints/latest.pt"):
        print("Found legacy checkpoint: checkpoints/latest.pt")
        # We need to peek at it to get the iteration number, or just guess
        try:
            # We wrap this in a try-except block just in case
            ckpt = torch.load("checkpoints/latest.pt", map_location="cpu")
            if isinstance(ckpt, dict) and 'iteration' in ckpt:
                return (ckpt['iteration'], "checkpoints/latest.pt")
        except:
            pass
            
    return None

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model, optimizer state, and replay buffer from checkpoint"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Load to CPU first to keep ReplayBuffer on CPU (save VRAM)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model weights (handling device mismatch)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state
    # We need to move optimizer state to device since we loaded to CPU
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    # Load replay buffer if available
    replay_buffer = ReplayBuffer(max_size=50000)
    if 'replay_buffer' in checkpoint:
        # Handle list format (correct)
        if isinstance(checkpoint['replay_buffer'], list):
            replay_buffer.buffer.extend(checkpoint['replay_buffer'])
            print(f"  Restored replay buffer with {len(replay_buffer)} examples")
        # Handle potential dict format (legacy/incorrect)
        elif isinstance(checkpoint['replay_buffer'], dict):
            print("  Warning: Skipping incompatible replay buffer in checkpoint")
    
    iteration = checkpoint.get('iteration', 0)
    
    print(f"  Resumed from iteration {iteration}")
    print(f"  Model and optimizer state restored")
    
    return iteration, replay_buffer

def save_checkpoint(model, optimizer, replay_buffer, iteration, filename):
    """Save complete checkpoint including model, optimizer, and replay buffer"""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer': list(replay_buffer.buffer)
    }
    torch.save(checkpoint, filename)
    print(f"  Saved checkpoint: {filename}")

def main():
    print("=" * 70)
    print("ECHELON TRAINING WITH C++ BACKEND")
    print("=" * 70)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EchelonNet(in_channels=13, num_res_blocks=5, num_filters=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(max_size=50000)
    
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check for existing checkpoint to resume from
    start_iteration = 0
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        iter_num, ckpt_path = latest_checkpoint
        start_iteration, replay_buffer = load_checkpoint(ckpt_path, model, optimizer, device)
        start_iteration += 1  # Start from next iteration
        print(f"\n✓ Resuming training from iteration {start_iteration}")
    else:
        print("\n✓ Starting fresh training (no checkpoint found)")
    
    # Training loop
    iterations_to_run = 10
    target_iteration = start_iteration + iterations_to_run
    
    print(f"Goal: Run for {iterations_to_run} iterations (until iteration {target_iteration})")
    
    for iteration in range(start_iteration, target_iteration):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Self-play with C++ (FAST!)
        print("[1/2] Generating self-play games...")
        start = time.time()
        
        for game_num in range(5):  # 5 games per iteration
            examples = play_game_cpp(model, num_simulations=50, max_moves=200, device=device)
            replay_buffer.add_examples(examples)
            print(f"  Game {game_num + 1}/5: {len(examples)} positions")
        
        print(f"  Self-play time: {time.time() - start:.1f}s")
        print(f"  Buffer size: {len(replay_buffer)}")
        
        # Training
        print("[2/2] Training neural network...")
        start = time.time()
        loss = train_iteration(model, optimizer, replay_buffer, batch_size=32, num_batches=50)
        print(f"  Loss: {loss:.4f}")
        print(f"  Training time: {time.time() - start:.1f}s")
        
        # Save checkpoint
        checkpoint_name = f"checkpoint_iter_{iteration + 1}.pt"
        save_checkpoint(model, optimizer, replay_buffer, iteration + 1, checkpoint_name)


    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
