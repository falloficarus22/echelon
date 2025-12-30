#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np

# Add C++ backend to path
sys.path.append(os.path.abspath("./cpp"))
import echelon_cpp

from model import EchelonNet
from play import find_latest_checkpoint

# UCI Protocol Handler for Echelon
# This allows the engine to be used with GUIs (Arena, CuteChess) and Lichess-bot.

class FastModelWrapper:
    """Wrapper to bridge PyTorch model with C++ MCTS"""
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
    
    def predict(self, board_tensor):
        tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, policy_logits = self.model(tensor)
        return value.item(), policy_logits.squeeze(0).cpu().numpy()

class UCIEngine:
    def __init__(self):
        self.model = None
        self.wrapper = None
        self.board = echelon_cpp.BoardState()
        self.mcts = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.param_num_blocks = 5 # Default, should match loaded model
        self.dummy_init()

    def dummy_init(self):
        """Initialize with a dummy model until weights are loaded"""
        self.model = EchelonNet(num_res_blocks=self.param_num_blocks).to(self.device)
        self.model.eval()
        self.wrapper = FastModelWrapper(self.model, self.device)
        self.mcts = echelon_cpp.MCTS(num_simulations=800) # Default sims

    def load_weights(self, path):
        """Load weights from a checkpoint"""
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device)
            # Try to detect architecture from checkpoint or just try loading
            # For now assume the architecture matches constants
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return True
        except Exception as e:
            sys.stderr.write(f"Error loading weights: {e}\n")
            return False

    def handle_uci(self):
        print("id name Echelon Zero")
        print("id author Antigravity")
        print("option name Hash type spin default 64 min 1 max 1024")
        print("option name Threads type spin default 1 min 1 max 1")
        print("option name NetworkFile type string default <autodiscover>")
        print("uciok")

    def handle_isready(self):
        print("readyok")

    def handle_ucinewgame(self):
        self.board = echelon_cpp.BoardState()
        self.board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        # Reset MCTS tree if supported, effectively new MCTS instance
        self.mcts = echelon_cpp.MCTS(num_simulations=800)

    def handle_position(self, args):
        # args example: "startpos moves e2e4 e7e5" or "fen ... moves ..."
        if not args:
            return

        parts = args.split()
        moves_idx = -1
        
        if parts[0] == "startpos":
            self.board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            if "moves" in parts:
                moves_idx = parts.index("moves")
        elif parts[0] == "fen":
            # Reconstruct FEN string. It usually stops at "moves"
            if "moves" in parts:
                moves_idx = parts.index("moves")
                fen_parts = parts[1:moves_idx]
            else:
                fen_parts = parts[1:]
            fen_str = " ".join(fen_parts)
            self.board.parse_fen(fen_str)
        
        # Apply moves
        if moves_idx != -1 and moves_idx + 1 < len(parts):
            moves_to_play = parts[moves_idx+1:]
            for move_str in moves_to_play:
                # Need to find the move in legal moves to get the object/integer needed by C++ backend
                # C++ board.make_move takes a move object/struct.
                # We need to map string (e.g. "e2e4") to a legal move.
                legal_moves = self.board.generate_legal_moves()
                found = False
                for m in legal_moves:
                    # Properties exposed in echelon_cpp_wrapper.cpp: source, target, flag
                    m_from = m.source
                    m_to = m.target
                    
                    # Helper to string
                    files = "abcdefgh"
                    f_str = files[m_from % 8] + str(m_from // 8 + 1)
                    t_str = files[m_to % 8] + str(m_to // 8 + 1)
                    
                    # Decode candidate string
                    cand_str = f_str + t_str
                    
                    # Check flag for promotion
                    # We can compare with the integer values from constants or just check blindly
                    # PROMO_Q is 4, R=3, B=2, N=1 based on board.cpp logic inferred earlier, 
                    # but let's check wrapper enum values if possible.
                    # Wrapper exports values:
                    # PROMOTION_QUEEN = 4 (default)
                    
                    # C++ Enum handling in pybind11 might return an object. 
                    # Casting to int usually works for comparison.
                    flag_val = int(m.flag)
                    
                    if flag_val == 4: cand_str += 'q'
                    elif flag_val == 3: cand_str += 'r'
                    elif flag_val == 2: cand_str += 'b'
                    elif flag_val == 1: cand_str += 'n'
                    
                    if cand_str == move_str:
                        self.board.make_move(m)
                        found = True
                        break
                
                if not found:
                    # Fallback for simple cases (non-promo) if string match failed closely
                    # e.g. "e2e4"
                    sys.stderr.write(f"Warning: could not apply move {move_str}\n")
                    pass

    def handle_go(self, args):
        # args example: "wtime 300000 btime 300000 winc 0 binc 0"
        # For now, we ignore time management and just search purely on nodes or fixed time (dummy)
        # Using fixed simulations is safest for a quick deploy.
        
        # Parse params if we want to be fancy
        # But just calling MCTS search is enough
        
        move_probs = self.mcts.search(self.board, self.wrapper)
        
        # Pick best move
        best_idx = max(move_probs, key=move_probs.get)
        
        # Decode index to UCI string
        from move_encoder import decode_index, index_to_square # We need these or similar
        # Since move_encoder is python, and board is C++, we have mismatch on "BoardState" object
        # decode_index expects a python BoardState to check legality/captures
        
        # We need a simpler way: iterate legal moves and find the one matching the Action Index.
        legal_moves = self.board.generate_legal_moves()
        best_move_obj = None
        
        for m in legal_moves:
            # We need to encode 'm' to get its index.
            # C++ MCTS has encode_move helper?
            # echelon_cpp.MCTS has encode_move(move) -> int
            if self.mcts.encode_move(m) == best_idx:
                best_move_obj = m
                break
        
        if best_move_obj:
            m = best_move_obj
            files = "abcdefgh"
            f_str = files[m.source % 8] + str(m.source // 8 + 1)
            t_str = files[m.target % 8] + str(m.target // 8 + 1)
            res = f_str + t_str
            
            flag_val = int(m.flag)
            if flag_val == 4: res += 'q'
            elif flag_val == 3: res += 'r'
            elif flag_val == 2: res += 'b'
            elif flag_val == 1: res += 'n'
            
            print(f"bestmove {res}")
        else:
            print("bestmove 0000") # Null/Resign

    def run(self):
        # Auto-load latest model at startup
        latest = find_latest_checkpoint()
        if latest:
            sys.stderr.write(f"Loading {latest}...\n")
            self.load_weights(latest)
        else:
            sys.stderr.write("Warning: No model found, playing randomly.\n")

        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                cmd = parts[0]
                args = " ".join(parts[1:])
                
                if cmd == "uci":
                    self.handle_uci()
                elif cmd == "isready":
                    self.handle_isready()
                elif cmd == "ucinewgame":
                    self.handle_ucinewgame()
                elif cmd == "position":
                    self.handle_position(args)
                elif cmd == "go":
                    self.handle_go(args)
                elif cmd == "stop":
                    pass # We are single-threaded blocking for now
                elif cmd == "quit":
                    break
                elif cmd == "setoption":
                    pass # TODO handling
                    
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
                
if __name__ == "__main__":
    # Unbuffered stdout for UCI communication
    sys.stdout.reconfigure(line_buffering=True)
    engine = UCIEngine()
    engine.run()
