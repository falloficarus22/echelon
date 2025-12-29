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

def play_game_cpp(model, num_simulations=100, max_moves=200):
    """Play one self-play game using C++ backend"""
    wrapper = FastModelWrapper(model)
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
    
    total_loss = 0
    for _ in range(num_batches):
        if len(replay_buffer) < batch_size:
            break
            
        positions, policies, values = replay_buffer.sample(batch_size)
        
        optimizer.zero_grad()
        pred_values, pred_policies = model(positions)
        
        value_loss = nn.MSELoss()(pred_values, values)
        policy_loss = -torch.mean(torch.sum(policies * nn.functional.log_softmax(pred_policies, dim=1), dim=1))
        
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches if num_batches > 0 else 0

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
    
    # Training loop
    for iteration in range(10):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Self-play with C++ (FAST!)
        print("[1/2] Generating self-play games...")
        start = time.time()
        
        for game_num in range(5):  # 5 games per iteration
            examples = play_game_cpp(model, num_simulations=50, max_moves=100)
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
        if (iteration + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint_iter_{iteration + 1}.pt")
            print(f"  Saved checkpoint")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
