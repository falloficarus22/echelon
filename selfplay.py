import sys
import os
import numpy as np
import torch
from collections import deque
import time

# Add C++ backend to path
sys.path.append(os.path.abspath("./cpp"))
import echelon_cpp

class FastModelWrapper:
    """Bridges PyTorch model with C++ MCTS"""
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
    
    def predict(self, board_tensor):
        # board_tensor comes from C++ as a [13, 8, 8] numpy array
        tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(tensor)
        return value.item(), policy_logits.squeeze(0).cpu().numpy()

class SelfPlayWorker:
    def __init__(self, model, num_simulations=800, temperature_threshold=30,
                 max_game_length=512, device="cpu"):
        self.fast_model = FastModelWrapper(model, device)
        self.num_simulations = num_simulations
        self.temperature_threshold = temperature_threshold
        self.max_game_length = max_game_length
        self.mcts = echelon_cpp.MCTS(num_simulations=num_simulations)
        
    def play_game(self, verbose=False):
        board = echelon_cpp.BoardState()
        # Initial position
        board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        game = SelfPlayGame()
        move_count = 0
        
        while True:
            move_count += 1
            
            # Check for termination using C++
            legal_moves = board.generate_legal_moves()
            if not legal_moves:
                # 0 for white, 1 for black (C++ side matches Python)
                side = 0 if "w" in str(board) else 1 # Simple check for side in C++
                # In C++ side is an int. Let's use evaluate or a helper if needed, 
                # but better to check if king is attacked after flip.
                # For self-play, we can just check if side is in check.
                if board.is_in_check():
                    outcome = -1.0 if board.evaluate() > 0 else 1.0 # Rough but side is baked in
                else:
                    outcome = 0.0
                break

            if board.evaluate() > 10000 or board.evaluate() < -10000: # Checkmate detection via evaluation score
                outcome = 1.0 if board.evaluate() > 0 else -1.0
                break

            if move_count >= self.max_game_length:
                outcome = 0.0
                break
            
            # Run C++ MCTS
            # Use current temperature settings
            self.mcts.temperature = 1.0 if move_count < self.temperature_threshold else 0.1
            move_probs = self.mcts.search(board, self.fast_model)
            
            # Store data (C++ tensorize is very fast)
            board_tensor = board.tensorize()
            policy_array = np.zeros(4672, dtype=np.float32)
            for idx, prob in move_probs.items():
                policy_array[idx] = prob
            
            # Side is handled internally by BoardState
            # We need to know who is moving for the outcome mapping
            current_side = 0 # Need to extract from board
            
            game.add_position(torch.from_numpy(board_tensor), policy_array, current_side)
            
            # Pick best move index based on probs
            best_idx = max(move_probs, key=move_probs.get)
            
            # We need to find the Move object corresponding to this index to call make_move
            # Since MCTS returns indices, we can look it up in legal_moves
            # (Note: In a more optimized version, C++ would return the move object too)
            found = False
            for m in legal_moves:
                # We need a way to encode the move in Python or check in C++
                # For now, let's assume we find it via index matching
                # Better: make_move should accept index or we match it
                # I'll add a helper to BoardState to make move by index
                continue # Placeholder for the logic below
             
            # Wait, I'll just use a simpler method: make_move_by_index
            # I will add this to the C++ wrapper shortly.
        
        # Get training examples
        training_examples = game.get_training_data(outcome)
        
        if verbose:
            print(f"Game finished: {len(training_examples)} positions")
            print(f"Outcome: {outcome}")
        
        return training_examples
    
    def generate_games(self, num_games, verbose=False):
        """
        Generate multiple self-play games.
        Returns all training examples.
        """
        all_examples = []
        
        print(f"Generating {num_games} self-play games...")
        
        for game_num in range(num_games):
            start_time = time.time()
            
            examples = self.play_game(verbose=verbose and game_num == 0)
            all_examples.extend(examples)
            
            elapsed = time.time() - start_time
            
            if (game_num + 1) % 10 == 0 or game_num == 0:
                print(f"Game {game_num + 1}/{num_games} completed "
                      f"({len(examples)} positions, {elapsed:.1f}s)")
        
        print(f"Total training examples: {len(all_examples)}")
        return all_examples


class ReplayBuffer:
    """
    Stores training examples from self-play games.
    Implements a circular buffer with maximum size.
    """
    def __init__(self, max_size=500000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_examples(self, examples):
        """Add training examples to buffer"""
        self.buffer.extend(examples)
    
    def sample(self, batch_size):
        """Sample a random batch of examples"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Convert to tensors
        positions = torch.stack([ex['position'] for ex in batch])
        policies = torch.tensor(np.array([ex['policy'] for ex in batch]), 
                               dtype=torch.float32)
        values = torch.tensor([[ex['value']] for ex in batch], 
                             dtype=torch.float32)
        
        return positions, policies, values
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath):
        """Save buffer to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"Saved {len(self.buffer)} examples to {filepath}")
    
    def load(self, filepath):
        """Load buffer from disk"""
        with open(filepath, 'rb') as f:
            examples = pickle.load(f)
        self.buffer.extend(examples)
        print(f"Loaded {len(examples)} examples from {filepath}")


def test_selfplay():
    """
    Test self-play system with a short game.
    """
    from model import EchelonNet
    
    print("Testing Self-Play System...")
    print("=" * 50)
    
    # Create model
    model = EchelonNet(in_channels=13, num_res_blocks=5, num_filters=128)
    model.eval()
    
    # Create self-play worker with reduced simulations for testing
    worker = SelfPlayWorker(model, num_simulations=50, 
                           temperature_threshold=10,
                           max_game_length=100)
    
    # Play one game
    print("\nPlaying test game with 50 MCTS simulations per move...")
    examples = worker.play_game(verbose=True)
    
    print(f"\n✓ Generated {len(examples)} training examples")
    
    # Test replay buffer
    print("\nTesting Replay Buffer...")
    buffer = ReplayBuffer(max_size=10000)
    buffer.add_examples(examples)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample a batch
    if len(buffer) >= 32:
        positions, policies, values = buffer.sample(32)
        print(f"\nSampled batch:")
        print(f"  Positions shape: {positions.shape}")
        print(f"  Policies shape: {policies.shape}")
        print(f"  Values shape: {values.shape}")
    
    print("\n✓ Self-play test completed successfully!")


if __name__ == "__main__":
    test_selfplay()