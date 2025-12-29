import numpy as np
from constants import *

class MoveEncoder():
    def __init__(self):
        self.num_moves = 4672 # 64 squares * 73 actions
        
        # Directions for Queen-like moves: N, NE, E, SE, S, SW, W, NW
        self.directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        
        # Knight moves
        self.knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
        
        # Underpromotion pieces
        self.under_pieces = [KNIGHT, BISHOP, ROOK]

    def encode_move(self, move):
        """
        Encode a chess move (from engine.py) to a policy head index (0-4671).
        AlphaZero style mapping: 73 actions per source square.
        """
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flag = (move >> 18) & 0x7
        
        from_rank, from_file = divmod(from_sq, 8)
        to_rank, to_file = divmod(to_sq, 8)
        
        dr = to_rank - from_rank
        df = to_file - from_file
        
        action_id = -1
        
        # 1. Underpromotions (Actions 64-72)
        if flag in [MOVE_FLAG_PROMOTION_KNIGHT, MOVE_FLAG_PROMOTION_BISHOP, MOVE_FLAG_PROMOTION_ROOK]:
            promo_map = {MOVE_FLAG_PROMOTION_KNIGHT: 0, MOVE_FLAG_PROMOTION_BISHOP: 1, MOVE_FLAG_PROMOTION_ROOK: 2}
            p_idx = promo_map[flag]
            # Relative file change for promotion: -1 (left), 0 (straight), 1 (right)
            # We assume White promotes on rank 7->8 and Black 2->1
            # But we can just use the 'to_file - from_file' 
            rel_file = df
            action_id = 64 + (rel_file + 1) * 3 + p_idx
            
        # 2. Knight moves (Actions 56-63)
        else:
            is_knight = False
            for i, (kdr, kdf) in enumerate(self.knight_moves):
                if dr == kdr and df == kdf:
                    action_id = 56 + i
                    is_knight = True
                    break
            
            # 3. Queen-like moves (Actions 0-55)
            if not is_knight:
                # Find direction
                if dr == 0 or df == 0 or abs(dr) == abs(df):
                    # Direction normalization
                    step_r = np.sign(dr)
                    step_f = np.sign(df)
                    dist = max(abs(dr), abs(df))
                    
                    for i, (ddr, ddf) in enumerate(self.directions):
                        if step_r == ddr and step_f == ddf:
                            action_id = i * 7 + (dist - 1)
                            break
        
        if action_id == -1:
            raise ValueError(f"Could not encode move {from_sq}->{to_sq} with flag {flag}")
            
        return from_sq * 73 + action_id

    def decode_index(self, index, board_state):
        """
        Decode a policy index back to a chess move integer.
        """
        from_sq = index // 73
        action_id = index % 73
        
        from_rank, from_file = divmod(from_sq, 8)
        
        to_sq = -1
        promo_type = 0 # 0 for Queen or None
        
        if action_id < 56: # Queen-like
            dir_idx = action_id // 7
            dist = (action_id % 7) + 1
            dr, df = self.directions[dir_idx]
            tr, tf = from_rank + dr * dist, from_file + df * dist
            if 0 <= tr < 8 and 0 <= tf < 8:
                to_sq = tr * 8 + tf
        
        elif action_id < 64: # Knight
            k_idx = action_id - 56
            dr, df = self.knight_moves[k_idx]
            tr, tf = from_rank + dr, from_file + df
            if 0 <= tr < 8 and 0 <= tf < 8:
                to_sq = tr * 8 + tf
                
        else: # Underpromotion
            u_idx = action_id - 64
            rel_file = (u_idx // 3) - 1
            p_idx = u_idx % 3
            promo_type = self.under_pieces[p_idx]
            
            # Target rank depends on side
            tr = 7 if board_state.side == WHITE else 0
            tf = from_file + rel_file
            if 0 <= tf < 8:
                to_sq = tr * 8 + tf

        if to_sq == -1:
            return None

        # Determine move components for the engine
        piece = board_state.get_piece_at_square(from_sq, board_state.side)
        if piece is None: return None
        
        captured = board_state.get_piece_at_square(to_sq, 1 - board_state.side)
        if captured is None: captured = 0
        
        flag = MOVE_FLAG_NORMAL
        
        # Handle special pawn moves
        if piece == PAWN:
            target_rank = to_sq // 8
            # Promotion
            if (board_state.side == WHITE and target_rank == 7) or (board_state.side == BLACK and target_rank == 0):
                if promo_type == KNIGHT: flag = MOVE_FLAG_PROMOTION_KNIGHT
                elif promo_type == BISHOP: flag = MOVE_FLAG_PROMOTION_BISHOP
                elif promo_type == ROOK: flag = MOVE_FLAG_PROMOTION_ROOK
                else: flag = MOVE_FLAG_PROMOTION_QUEEN
            # Double Push
            elif abs(to_sq - from_sq) == 16:
                flag = MOVE_FLAG_DOUBLE_PAWN_PUSH
            # En Passant
            elif to_sq == board_state.en_passant_sq:
                flag = MOVE_FLAG_EN_PASSANT
                captured = PAWN
                
        # Castling
        elif piece == KING and abs(to_sq % 8 - from_sq % 8) == 2:
            flag = MOVE_FLAG_CASTLING

        return board_state.encode_move(from_sq, to_sq, piece, captured, flag=flag)

    def create_policy_mask(self, legal_moves):
        mask = np.zeros(self.num_moves, dtype=np.float32)
        for move in legal_moves:
            mask[self.encode_move(move)] = 1.0
        return mask

    def create_policy_target(self, move_probs):
        target = np.zeros(self.num_moves, dtype=np.float32)
        for move, prob in move_probs.items():
            target[self.encode_move(move)] = prob
        return target

# Global instance
move_encoder = MoveEncoder()

def encode_move(move): return move_encoder.encode_move(move)
def decode_index(index, board_state): return move_encoder.decode_index(index, board_state)
def create_policy_mask(legal_moves): return move_encoder.create_policy_mask(legal_moves)
def create_policy_target(move_probs): return move_encoder.create_policy_target(move_probs)

if __name__ == "__main__":
    from engine import BoardState
    print("Testing 4672 Move Encoder...")
    board = BoardState()
    board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    legal_moves = board.generate_legal_moves(board.side)
    print(f"Testing {len(legal_moves)} moves...")
    
    for move in legal_moves:
        idx = encode_move(move)
        decoded = decode_index(idx, board)
        if decoded != move:
            print(f"Encoding Error! Move {move} != Decoded {decoded} (Index {idx})")
            exit(1)
    
    print("Success! All moves trip-coded correctly.")