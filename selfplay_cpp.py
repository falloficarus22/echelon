"""selfplay_cpp.py - Fast self-play using C++ backend"""
import sys
import os
import torch
import numpy as np

# Add C++ backend to path
sys.path.append(os.path.abspath("./cpp"))
try:
    import echelon_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("WARNING: C++ backend not available, falling back to slow Python")
    from selfplay import SelfPlayWorker
    from engine import BoardState

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FastModelWrapper:
    """Bridges PyTorch model with C++ MCTS"""
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Test the model output format once
        test_input = torch.zeros(1, 13, 8, 8).to(device)
        with torch.no_grad():
            out1, out2 = self.model(test_input)
        
        print(f"Model output shapes: out1={out1.shape}, out2={out2.shape}")
        
        # Determine which is value and which is policy
        if out1.shape[-1] == 1 or out1.numel() == 1:
            self.value_first = True
            print("✓ Model returns (value, policy)")
        elif out2.shape[-1] == 1 or out2.numel() == 1:
            self.value_first = False
            print("✓ Model returns (policy, value)")
        else:
            raise RuntimeError(f"Cannot determine output order from shapes: {out1.shape}, {out2.shape}")
    
    def predict(self, board_tensor):
        # board_tensor comes from C++ as [13, 8, 8] numpy array
        tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out1, out2 = self.model(tensor)
        
        if self.value_first:
            value = out1.squeeze().item()
            policy = out2.squeeze(0).cpu().numpy()
        else:
            policy = out1.squeeze(0).cpu().numpy()
            value = out2.squeeze().item()
        
        return value, policy


class CPPSelfPlayWorker:
    """Fast self-play using C++ backend (60x faster than Python)"""
    def __init__(self, model, num_simulations=400, temperature_threshold=30, max_game_length=512):
        self.model = model
        self.wrapper = FastModelWrapper(model, device=next(model.parameters()).device)
        self.num_simulations = int(num_simulations)
        self.temperature_threshold = int(temperature_threshold)
        self.max_game_length = int(max_game_length)
        
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ backend not available")
    
    def generate_games(self, num_games, verbose=False):
        """Generate self-play games using C++ engine"""
        all_examples = []
        
        for game_idx in range(int(num_games)):
            # Create new board for each game
            board = echelon_cpp.BoardState()
            board.parse_fen(START_FEN)
            
            # Create MCTS for this game
            mcts = echelon_cpp.MCTS(
                num_simulations=self.num_simulations,
                c_puct=1.5,
                temperature=1.0,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25
            )
            
            game_history = []  # [(position_tensor, policy_array, side)]
            move_count = 0
            
            while move_count < self.max_game_length:
                # Get position tensor
                pos_tensor = torch.from_numpy(board.tensorize()).float()
                
                # Adjust temperature (greedy near endgame)
                if move_count >= self.temperature_threshold:
                    mcts.set_temperature(0.1)
                else:
                    mcts.set_temperature(1.0)
                
                # Run MCTS search
                move_probs = mcts.search(board, self.wrapper)
                
                if not move_probs:
                    break
                
                # Convert move_probs dict to policy array [4672]
                policy_array = np.zeros(4672, dtype=np.float32)
                for move_idx, prob in move_probs.items():
                    policy_array[move_idx] = prob
                
                # Store (position, policy, side_to_move)
                current_side = move_count % 2  # 0=White, 1=Black
                game_history.append((pos_tensor, policy_array, current_side))
                
                # Select best move
                best_idx = max(move_probs, key=move_probs.get)
                
                # Find move object and play it
                legal_moves = board.generate_legal_moves()
                move_played = False
                
                for move in legal_moves:
                    if mcts.encode_move(move) == best_idx:
                        board.make_move(move)
                        move_played = True
                        break
                
                if not move_played:
                    # Fallback: play first legal move
                    if legal_moves:
                        board.make_move(legal_moves[0])
                    else:
                        break
                
                move_count += 1
                
                # Check for terminal state
                legal_moves = board.generate_legal_moves()
                if not legal_moves:
                    break
            
            # Determine game outcome
            result_value = 0.0
            winner = -1
            
            legal_after = board.generate_legal_moves()
            if not legal_after:
                if board.is_in_check():
                    # Current side lost (opponent won)
                    winner = 1 - (move_count % 2)
                    result_value = 1.0
                else:
                    # Stalemate
                    result_value = 0.0
            else:
                # Max length reached -> draw
                result_value = 0.0
            
            # Convert history to training examples
            for pos, policy, player_side in game_history:
                if result_value == 0.0:
                    value = 0.0  # Draw
                else:
                    # +1 if player won, -1 if player lost
                    value = 1.0 if (winner == player_side) else -1.0
                
                all_examples.append((pos, policy, value))
            
            if verbose:
                outcome = "Draw" if result_value == 0.0 else f"{'White' if winner == 0 else 'Black'} wins"
                print(f"Game {game_idx+1}/{num_games}: {move_count} moves, {len(game_history)} positions, {outcome}")
        
        return all_examples


# Export the appropriate worker
if CPP_AVAILABLE:
    print("✓ Using C++ accelerated self-play (60x faster)")
    SelfPlayWorker = CPPSelfPlayWorker
else:
    print("⚠ Using slow Python self-play")
    # SelfPlayWorker already imported from selfplay.py above