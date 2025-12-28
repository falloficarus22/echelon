import numpy as np
from constants import *

class MoveEncoder():
    def __init__(self):
        self.num_moves = 4672

        # Promotion pieces (excluding queen which is treated as normal move)
        self.promotion_pieces = [KNIGHT, BISHOP, ROOK]
        self._build_move_lookup()

    def _build_move_lookup(self):
        """
        Build lookup tables for fast encoding/decoding
        """
        # Map move -> index
        self.move_to_index = {}

        # Map index -> move
        self.index_to_move = {}
        index = 0

        for from_sq in range(64):
            for to_sq in range(64):
                # Store mapping
                key = (from_sq, to_sq, 0)
                self.move_to_index[key] = index
                self.index_to_move[index] = key
                index += 1

        assert index == self.num_moves, f"Expected {self.num_moves}, got {index}"

    def encode_move(self, move):
        """
        Encode a chess move (from engine.py) to policy head index
        """
        # Decode the move
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        piece = (move >> 12) & 0x7
        flag = (move >> 18) & 0x7

        # Determine promotion type
        promo_type = 0 # Default: for normal move or queen promotion
        promotions = [MOVE_FLAG_PROMOTION_KNIGHT,
                      MOVE_FLAG_PROMOTION_BISHOP,
                      MOVE_FLAG_PROMOTION_ROOK]
        
        if flag in promotions:
            # Underpromotion
            promo_map = {
                MOVE_FLAG_PROMOTION_KNIGHT: KNIGHT,
                MOVE_FLAG_PROMOTION_BISHOP: BISHOP,
                MOVE_FLAG_PROMOTION_ROOK: ROOK
            }
            promo_type = promo_map[flag]

        # Look up index
        key = (from_sq, to_sq, promo_type)
        return self.move_to_index[key]
    
    def decode_index(self, index, board_state):
        """Decode a policy index back to a chess move"""
        if index not in self.index_to_move:
            return None
        
        from_sq, to_sq, promo_type = self.index_to_move[index]

        # Determine the piece being moved
        piece = board_state.get_piece_at_square(from_sq, board_state.side)
        if piece is None:
            return None
        
        # Determine the captured piece
        captured = board_state.get_piece_at_square(to_sq, 1 - board_state.side)
        if captured is None:
            captured = 0

        # Determine move flag
        flag = MOVE_FLAG_NORMAL

        # Check for special moves
        if piece == PAWN:
            # Check for promotion
            target_rank = to_sq // 8
            
            if (board_state.side == WHITE and target_rank == 7) or (board_state.side == BLACK and target_rank == 0):
                # Promotion move
                if promo_type == KNIGHT:
                    flag = MOVE_FLAG_PROMOTION_KNIGHT
                elif promo_type == BISHOP:
                    flag = MOVE_FLAG_PROMOTION_BISHOP
                elif promo_type == ROOK:
                    flag = MOVE_FLAG_PROMOTION_ROOK
                else:
                    flag == MOVE_FLAG_PROMOTION_QUEEN
                
            # Check for double pawn push
            elif abs(from_sq - to_sq) == 16:
                flag = MOVE_FLAG_DOUBLE_PAWN_PUSH

            elif to_sq == board_state.en_passant_sq:
                flag = MOVE_FLAG_EN_PASSANT
                captured = PAWN

        # Check for castling
        elif piece == KING and abs(from_sq - to_sq) == 2:
            flag = MOVE_FLAG_CASTLING

        # Encode the move
        move = board_state.encode_move(
            source=from_sq,
            target=to_sq,
            piece=piece,
            captured=captured,
            flag=flag
        )

        return move
    
    def encode_legal_moves(self, legal_moves):
        """
        Encode a list of legal moves to policy indices.
        """
        return [self.encode_move(move) for move in legal_moves]
    
    def create_policy_mask(self, legal_moves):
        """
        Create a mask for legal moves (used in MCTS).
        """
        mask = np.zeros(self.num_moves, dtype=np.float32)
        legal_indices = self.encode_legal_moves(legal_moves)
        mask[legal_indices] = 1.0
        return mask
    
    def create_policy_target(self, move_probs):
        """
        Create a policy target vector from move probabilities.
        """
        target = np.zeros(self.num_moves, dtype=np.float32)
        
        total_prob = 0.0
        for move, prob in move_probs.items():
            index = self.encode_move(move)
            target[index] = prob
            total_prob += prob
        
        # Normalize to ensure it sums to 1
        if total_prob > 0:
            target /= total_prob
        
        return target


# Global encoder instance
move_encoder = MoveEncoder()

# Convenience functions

def encode_move(move):
    """Encode a single move to policy index."""
    return move_encoder.encode_move(move)


def decode_index(index, board_state):
    """Decode a policy index to a move."""
    return move_encoder.decode_index(index, board_state)


def encode_legal_moves(legal_moves):
    """Encode a list of legal moves."""
    return move_encoder.encode_legal_moves(legal_moves)


def create_policy_mask(legal_moves):
    """Create a legal move mask."""
    return move_encoder.create_policy_mask(legal_moves)


def create_policy_target(move_probs):
    """Create a policy target vector."""
    return move_encoder.create_policy_target(move_probs)


# Testing

def test_move_encoding():
    """Test the move encoding system."""
    from engine import BoardState
    
    print("Testing Move Encoding System...")
    print("=" * 60)
    
    # Create a board
    board = BoardState()
    board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Generate legal moves
    legal_moves = board.generate_all_moves(board.side)
    print(f"Generated {len(legal_moves)} legal moves")
    
    # Test encoding/decoding
    print("\nTesting encode/decode round-trip...")
    errors = 0
    for move in legal_moves:
        # Encode
        index = encode_move(move)
        
        # Decode
        decoded_move = decode_index(index, board)
        
        # Check if we get back the same move
        if decoded_move != move:
            errors += 1
            print(f"  ERROR: Move {move} → index {index} → move {decoded_move}")
    
    if errors == 0:
        print(f"  All {len(legal_moves)} moves encoded/decoded correctly!")
    else:
        print(f"  {errors} errors found!")
    
    # Test policy mask
    print("\nTesting policy mask creation...")
    mask = create_policy_mask(legal_moves)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Legal moves marked: {mask.sum():.0f} (expected: {len(legal_moves)})")
    print(f"  Illegal moves: {(mask == 0).sum():.0f}")
    
    # Test with a few sample moves
    print("\nSample move encodings:")
    for i, move in enumerate(legal_moves[:5]):
        decoded = board.decode_move(move)
        index = encode_move(move)
        from_sq = decoded['from']
        to_sq = decoded['to']
        from_file = chr(ord('a') + from_sq % 8)
        from_rank = from_sq // 8 + 1
        to_file = chr(ord('a') + to_sq % 8)
        to_rank = to_sq // 8 + 1
        print(f"  {from_file}{from_rank}{to_file}{to_rank} → index {index}")
    
    # Test policy target
    print("\nTesting policy target creation...")
    move_probs = {legal_moves[0]: 0.5, legal_moves[1]: 0.3, legal_moves[2]: 0.2}
    target = create_policy_target(move_probs)
    print(f"  Target shape: {target.shape}")
    print(f"  Target sum: {target.sum():.3f} (should be 1.0)")
    print(f"  Non-zero entries: {(target > 0).sum()}")
    print(" Move encoding tests complete!")
    
    return move_encoder


if __name__ == "__main__":
    encoder = test_move_encoding()