import numpy as np
import torch
from collections import deque
from engine import BoardState
from mcts import MCTS
from move_encoder import encode_move, create_policy_target
import pickle
import time


class SelfPlayGame:
    """
    Represents a single self-play game.
    Stores positions, policies, and outcomes for training.
    """
    def __init__(self):
        self.positions = []  # Board tensors
        self.policies = []   # MCTS visit probabilities
        self.current_player = []  # Side to move
        
    def add_position(self, board_tensor, policy, side):
        """Add a position from the game"""
        self.positions.append(board_tensor)
        self.policies.append(policy)
        self.current_player.append(side)
    
    def get_training_data(self, outcome):
        """
        Convert game to training examples.
        outcome: 1.0 for white win, -1.0 for black win, 0.0 for draw
        """
        training_examples = []
        
        for i, (pos, policy, player) in enumerate(zip(
            self.positions, self.policies, self.current_player
        )):
            # Value from perspective of current player
            if outcome == 0:
                value = 0.0  # Draw
            elif (outcome > 0 and player == 0) or (outcome < 0 and player == 1):
                value = 1.0  # Win
            else:
                value = -1.0  # Loss
            
            training_examples.append({
                'position': pos,
                'policy': policy,
                'value': value
            })
        
        return training_examples


class SelfPlayWorker:
    """
    Generates self-play games for training data.
    """
    def __init__(self, model, num_simulations=800, temperature_threshold=30,
                 max_game_length=512):
        """
        Args:
            model: Neural network for MCTS
            num_simulations: MCTS simulations per move
            temperature_threshold: Move number to switch from stochastic to deterministic
            max_game_length: Maximum moves before declaring draw
        """
        self.model = model
        self.num_simulations = num_simulations
        self.temperature_threshold = temperature_threshold
        self.max_game_length = max_game_length
        
    def play_game(self, verbose=False):
        """
        Play a single self-play game.
        Returns training examples from the game.
        """
        board = BoardState()
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board.parse_fen(start_fen)
        
        game = SelfPlayGame()
        move_count = 0
        
        if verbose:
            print("Starting self-play game...")
        
        while True:
            move_count += 1
            
            # Check for game termination
            legal_moves = board.generate_legal_moves(board.side)
            
            if len(legal_moves) == 0:
                # Game over
                if board.is_in_check(board.side):
                    # Checkmate - opponent wins
                    outcome = -1.0 if board.side == 0 else 1.0
                    if verbose:
                        print(f"Checkmate! {'Black' if board.side == 0 else 'White'} wins")
                else:
                    # Stalemate
                    outcome = 0.0
                    if verbose:
                        print("Stalemate!")
                break
            
            # Check for draw by repetition or 50-move rule
            if board.halfmove_clock >= 100:  # 50 moves
                outcome = 0.0
                if verbose:
                    print("Draw by 50-move rule")
                break
            
            # Check for maximum game length
            if move_count >= self.max_game_length:
                outcome = 0.0
                if verbose:
                    print(f"Draw by max length ({self.max_game_length} moves)")
                break
            
            # Determine temperature for this move
            # Use temperature=1 for first N moves, then temperature=0 (deterministic)
            if move_count < self.temperature_threshold:
                temperature = 1.0
            else:
                temperature = 0.1  # Near-deterministic
            
            # Run MCTS
            mcts = MCTS(self.model, num_simulations=self.num_simulations, 
                       temperature=temperature)
            best_move, move_probs = mcts.search(board)
            
            if best_move is None:
                if verbose:
                    print("MCTS failed to find a move (possible engine/board corruption)")
                # Force draw or break
                outcome = 0.0
                break
            
            # Store position and policy
            board_tensor = board.tensorize_board()
            
            # Convert move_probs dict to array format for training
            policy_array = np.zeros(4672, dtype=np.float32)
            for move, prob in move_probs.items():
                try:
                    move_idx = encode_move(move)
                    policy_array[move_idx] = prob
                except (ValueError, IndexError):
                    continue
            
            game.add_position(board_tensor, policy_array, board.side)
            
            # Make the move
            if verbose and move_count <= 10:
                decoded = board.decode_move(best_move)
                from_sq = decoded['from']
                to_sq = decoded['to']
                move_str = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
                move_str += f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
                print(f"Move {move_count}: {move_str}")
            
            board.make_move(best_move)
        
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