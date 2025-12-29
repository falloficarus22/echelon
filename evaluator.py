import numpy as np
import time
import torch
from engine import BoardState
from mcts import MCTS
from move_encoder import encode_move

class Evaluator:
    """
    Evaluates the strength of a model by playing games against a baseline.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def play_match(self, num_games=10, mcts_simulations=100):
        """
        Play a match between the Neural Network (MCTS) and Greedy Baseline.
        """
        results = {'win': 0, 'loss': 0, 'draw': 0}
        
        for i in range(num_games):
            # Alternate sides: NN is white on even games, black on odd
            nn_side = 0 if i % 2 == 0 else 1
            
            outcome = self.play_single_game(nn_side, mcts_simulations)
            
            # Outcome is 1.0 if White wins, -1.0 if Black wins, 0.0 for draw
            if outcome == 0:
                results['draw'] += 1
            elif (outcome == 1.0 and nn_side == 0) or (outcome == -1.0 and nn_side == 1):
                results['win'] += 1
            else:
                results['loss'] += 1
                
        # Calculate Win Rate
        win_rate = (results['win'] + 0.5 * results['draw']) / num_games
        
        # Calculate relative Elo (Greedy Baseline = 500)
        # Formula: Elo = Baseline + 400 * log10(win_rate / (1 - win_rate))
        if win_rate == 1.0:
            elo = 1200 # Cap for small sample size
        elif win_rate == 0.0:
            elo = 0
        else:
            elo = 500 + 400 * np.log10(win_rate / (1 - win_rate))
            
        return elo, results

    def play_single_game(self, nn_side, simulations):
        board = BoardState()
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board.parse_fen(start_fen)
        
        move_count = 0
        max_moves = 200
        
        while move_count < max_moves:
            legal_moves = board.generate_legal_moves(board.side)
            if not legal_moves:
                if board.is_in_check(board.side):
                    # Checkmate - return 1.0 if black was to move (white wins) and vice versa
                    return -1.0 if board.side == 0 else 1.0
                return 0.0 # Stalemate
            
            if board.halfmove_clock >= 100:
                return 0.0 # 50-move rule
            
            if board.side == nn_side:
                # NN chooses move using MCTS
                mcts = MCTS(self.model, num_simulations=simulations, temperature=0.1)
                best_move, _ = mcts.search(board)
                if best_move is None: return 0.0
            else:
                # Baseline chooses move greedily
                best_move = board.get_greedy_move()
                if best_move is None: return 0.0
                
            board.make_move(best_move)
            move_count += 1
            
        return 0.0 # Max moves reached
