#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np

# Add C++ backend to path
sys.path.insert(0, os.path.abspath("./cpp"))

# Try to import and initialize C++ module
try:
    import echelon_cpp
    echelon_cpp.init()  # CRITICAL: Initialize attack tables
    sys.stderr.write("✓ C++ module loaded and initialized\n")
except ImportError as e:
    sys.stderr.write(f"ERROR: Cannot import echelon_cpp: {e}\n")
    sys.stderr.write("Please run: ./build_cpp.sh\n")
    sys.exit(1)
except Exception as e:
    sys.stderr.write(f"ERROR: Failed to initialize C++ module: {e}\n")
    sys.exit(1)

from model import EchelonNet
from play import find_latest_checkpoint


class FastModelWrapper:
    """Wrapper to bridge PyTorch model with C++ MCTS"""
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict(self, board_tensor):
        tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, policy_logits = self.model(tensor)
        return value.item(), policy_logits.squeeze(0).cpu().numpy()


class UCIEngine:
    def __init__(self):
        self.model = None
        self.wrapper = None
        self.board = None
        self.mcts = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.param_num_blocks = 5 
        self.initialized = False

    def lazy_init(self):
        """Heavy lifting happens only when the GUI/Lichess asks 'isready'"""
        if self.initialized:
            return
        
        sys.stderr.write("Loading Engine Components...\n")
        sys.stderr.flush()
        
        try:
            # 1. Initialize C++ BoardState
            self.board = echelon_cpp.BoardState()
            self.board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            
            # 2. Initialize Model and Load Weights
            self.model = EchelonNet(num_res_blocks=self.param_num_blocks).to(self.device)
            latest = find_latest_checkpoint()
            if latest:
                sys.stderr.write(f"Loading weights from: {latest}\n")
                self.load_weights(latest)
            else:
                sys.stderr.write("Warning: No weights found! Using random initialization.\n")
            
            self.model.eval()
            self.wrapper = FastModelWrapper(self.model, self.device)
            
            # 3. Initialize MCTS
            self.mcts = echelon_cpp.MCTS(
                num_simulations=800,
                c_puct=1.5,
                temperature=1.0,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25
            )
            
            self.initialized = True
            sys.stderr.write("✓ Initialization Complete. Engine Ready.\n")
            sys.stderr.flush()
            
        except Exception as e:
            sys.stderr.write(f"ERROR during initialization: {e}\n")
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise

    def load_weights(self, path):
        if not os.path.exists(path):
            sys.stderr.write(f"Warning: Checkpoint not found: {path}\n")
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return True
        except Exception as e:
            sys.stderr.write(f"Error loading weights: {e}\n")
            return False

    def handle_uci(self):
        print("id name Echelon Zero")
        print("id author Antigravity")
        print("option name Move Overhead type spin default 100 min 0 max 5000")
        print("option name Hash type spin default 256 min 1 max 1024")
        print("option name Threads type spin default 1 min 1 max 1")
        print("uciok")
        sys.stdout.flush()

    def handle_isready(self):
        try:
            self.lazy_init()
            print("readyok")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"ERROR in isready: {e}\n")
            sys.stderr.flush()

    def handle_ucinewgame(self):
        if not self.initialized: 
            self.lazy_init()
        self.board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def handle_position(self, args):
        if not self.initialized: 
            self.lazy_init()
        if not args: 
            return

        parts = args.split()
        moves_idx = -1
        
        if parts[0] == "startpos":
            self.board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            if "moves" in parts:
                moves_idx = parts.index("moves")
        elif parts[0] == "fen":
            if "moves" in parts:
                moves_idx = parts.index("moves")
                fen_parts = parts[1:moves_idx]
            else:
                fen_parts = parts[1:]
            fen_str = " ".join(fen_parts)
            self.board.parse_fen(fen_str)
        
        if moves_idx != -1 and moves_idx + 1 < len(parts):
            moves_to_play = parts[moves_idx+1:]
            for move_str in moves_to_play:
                legal_moves = self.board.generate_legal_moves()
                found = False
                for m in legal_moves:
                    # Use correct property names from C++ binding
                    m_from = m.from_sq  # Changed from m.source
                    m_to = m.to_sq      # Changed from m.target
                    
                    files = "abcdefgh"
                    f_str = files[m_from % 8] + str(m_from // 8 + 1)
                    t_str = files[m_to % 8] + str(m_to // 8 + 1)
                    cand_str = f_str + t_str
                    
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
                    sys.stderr.write(f"Warning: could not apply move {move_str}\n")
                    sys.stderr.flush()

    def handle_go(self, args):
        if not self.initialized: 
            self.lazy_init()
        
        try:
            # MCTS search
            move_probs = self.mcts.search(self.board, self.wrapper)
            
            if not move_probs:
                print("bestmove 0000")
                sys.stdout.flush()
                return
            
            best_idx = max(move_probs, key=move_probs.get)
            
            legal_moves = self.board.generate_legal_moves()
            best_move_obj = None
            
            for m in legal_moves:
                if self.mcts.encode_move(m) == best_idx:
                    best_move_obj = m
                    break
            
            if best_move_obj:
                m = best_move_obj
                files = "abcdefgh"
                f_str = files[m.from_sq % 8] + str(m.from_sq // 8 + 1)
                t_str = files[m.to_sq % 8] + str(m.to_sq // 8 + 1)
                res = f_str + t_str
                
                flag_val = int(m.flag)
                if flag_val == 4: res += 'q'
                elif flag_val == 3: res += 'r'
                elif flag_val == 2: res += 'b'
                elif flag_val == 1: res += 'n'
                
                print(f"bestmove {res}")
            else:
                print("bestmove 0000")
            
            sys.stdout.flush()
            
        except Exception as e:
            sys.stderr.write(f"ERROR in go: {e}\n")
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            print("bestmove 0000")
            sys.stdout.flush()

    def run(self):
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
                args = " ".join(parts[1:]) if len(parts) > 1 else ""
                
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
                elif cmd == "quit":
                    break
                else:
                    sys.stderr.write(f"Unknown command: {cmd}\n")
                    sys.stderr.flush()
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                sys.stderr.write(f"UCI Error: {e}\n")
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()


if __name__ == "__main__":
    # Unbuffered I/O for UCI communication
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    engine = UCIEngine()
    engine.run()
    