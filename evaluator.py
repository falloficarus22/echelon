import numpy as np
import time
import torch
import sys
import os

# Add C++ backend to path
sys.path.append(os.path.abspath("./cpp"))
try:
    import echelon_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: echelon_cpp not found, falling back to slow Python implementation")
    from engine import BoardState
    from mcts import MCTS

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

class Evaluator:
    """
    Evaluates the strength of a model by playing games against a baseline.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        if CPP_AVAILABLE:
            self.fast_model = FastModelWrapper(model, device)
        
    def play_match(self, num_games=10, mcts_simulations=100):
        """
        Play a match between the Neural Network (MCTS) and Greedy Baseline.
        """
        results = {'win': 0, 'loss': 0, 'draw': 0}
        
        start_time = time.time()
        for i in range(num_games):
            # Alternate sides: NN is white on even games, black on odd
            nn_side = 0 if i % 2 == 0 else 1
            
            if CPP_AVAILABLE:
                outcome = self.play_single_game_cpp(nn_side, mcts_simulations)
            else:
                outcome = self.play_single_game_python(nn_side, mcts_simulations)
            
            # Outcome is 1.0 if White wins, -1.0 if Black wins, 0.0 for draw
            if outcome == 0:
                results['draw'] += 1
            elif (outcome == 1.0 and nn_side == 0) or (outcome == -1.0 and nn_side == 1):
                results['win'] += 1
            else:
                results['loss'] += 1
            
            if (i + 1) % 5 == 0:
                print(f"Played {i+1}/{num_games} games...")
                
        # Calculate Win Rate
        win_rate = (results['win'] + 0.5 * results['draw']) / num_games
        
        # Calculate relative Elo (Greedy Baseline = 500)
        if win_rate == 1.0:
            elo = 1200 # Cap for small sample size
        elif win_rate == 0.0:
            elo = 0
        else:
            elo = 500 + 400 * np.log10(win_rate / (1 - win_rate))
            
        print(f"Match finished in {time.time() - start_time:.1f}s")
        return elo, results

    def play_single_game_cpp(self, nn_side, simulations):
        board = echelon_cpp.BoardState()
        board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        mcts = echelon_cpp.MCTS(num_simulations=simulations)
        move_count = 0
        max_moves = 200
        
        while move_count < max_moves:
            legal_moves = board.generate_legal_moves()
            if not legal_moves:
                if board.is_in_check():
                    # If current side is in check and no moves, they lost.
                    # Current side (board.side) is the loser.
                    # board.side in CWrapper isn't directly exposed as 0/1 usually, 
                    # but we can infer outcome. 
                    # Actually, simply: if it's white's turn (0) and checkmate, Black (1) wins (-1.0)
                    # We can check 'w' in FEN string if side is not exposed
                    is_white = "w" in board.to_fen().split(" ")[1]
                    return -1.0 if is_white else 1.0
                return 0.0 # Stalemate
            
            # Determine whose turn it is
            is_white = "w" in board.to_fen().split(" ")[1]
            current_side = 0 if is_white else 1
            
            best_move = None
            
            if current_side == nn_side:
                # NN chooses move using MCTS
                mcts.temperature = 0.1 # Low temp for play
                move_probs = mcts.search(board, self.fast_model)
                best_idx = max(move_probs, key=move_probs.get)
                
                # Find move object
                for m in legal_moves:
                    if mcts.encode_move(m) == best_idx:
                        best_move = m
                        break
                # Fallback if mapped move not found (rare)
                if best_move is None: best_move = legal_moves[0]
            else:
                # Greedy Baseline using C++ evaluation
                best_score = -float('inf')
                
                # Simple greedy: Make move, evaluate from opponent perspective, negate
                for m in legal_moves:
                    # Clone board to test move
                    # board.copy() might not be available, but we can make/unmake if supported
                    # or parse FEN. Parsing FEN is safest if copy not exposed.
                    fen = board.to_fen()
                    temp_board = echelon_cpp.BoardState()
                    temp_board.parse_fen(fen)
                    
                    temp_board.make_move(m)
                    
                    # Evaluate for the side that just moved is roughly -1 * eval for new side
                    # C++ evaluate() usually returns score relative to side to move
                    # So if I just moved, now it's opponent turn. 
                    # Opponent wants to maximize their score.
                    # So I want to minimize their score.
                    # Or simpler: evaluate() returns static score (positive = good for white)
                    # Let's assume static evaluation (White +, Black -)
                    score = temp_board.evaluate()
                    
                    # If I am black (1), I want minimal score. If White (0), max.
                    if current_side == 1: # Black
                        score = -score
                        
                    if score > best_score:
                        best_score = score
                        best_move = m
                        
            board.make_move(best_move)
            move_count += 1
            
        return 0.0 # Draw

    def play_single_game_python(self, nn_side, simulations):
        # Legacy Python implementation
        board = BoardState()
        board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        move_count = 0
        while move_count < 200:
            legal_moves = board.generate_legal_moves(board.side)
            if not legal_moves:
                if board.is_in_check(board.side):
                    return -1.0 if board.side == 0 else 1.0
                return 0.0
            
            if board.side == nn_side:
                mcts = MCTS(self.model, num_simulations=simulations, temperature=0.1)
                best_move, _ = mcts.search(board)
            else:
                best_move = board.get_greedy_move()
                
            if best_move is None: return 0.0
            board.make_move(best_move)
            move_count += 1
        return 0.0
