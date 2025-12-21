import numpy as np
from constants import *
from attacks import king_attacks_table, knight_attacks_table, pawn_attacks_table

class BoardState:
    def __init__(self):
        """
        Initialize a new board state.
        # Used np.uint64 instead of python integers to optimize bitwise operations
        """
        # Bitboards: 12 integers (6 piece types * 2 colors)
        # We initialize the bitboard to 0 and assign the intgers later
        self.bitboards = np.zeros(12, dtype = np.uint64)

        # Occupancies: 3 integers (White, Black, All)
        # These are "helper" bitboards used to check if a square is empty
        self.occupancies = np.zeros(3, dtype = np.uint64)

        # Game State Variables
        self.side = WHITE
        self.en_passant_sq = -1 # -1 means no en-passant square available
        self.castle_rights = 0

    def parse_fen(self, fen):
        """
        Resets the board and loads the state from a FEN string.
        """
        # Clear everything first
        self.bitboards.fill(0)
        self.occupancies.fill(0)

        # Split the FEN string into components
        parts = fen.split()
        board_part = parts[0]
        turn_part = parts[1]
        castle_part = parts[2]

        # Parse board layout
        rank = 7
        file = 0

        piece_map = {
            'P': PAWN, 'N': KNIGHT, 'B':BISHOP, 'R': ROOK, 'Q': QUEEN, 'K': KING,
            'p': PAWN, 'n': KNIGHT, 'b':BISHOP, 'r': ROOK, 'q': QUEEN, 'k': KING
        }

        for char in board_part:
            if char =="/":
                rank -= 1
                file = 0
            elif char.isdigit():
                # Digits mean empty squares
                empty_skips = int(char)
                file += empty_skips
            else:
                # Actual pieces
                piece_type = piece_map[char]
                color = WHITE if char.isupper() else BLACK

                # Global piece index
                index = piece_type + (color * 6)
                square = rank * 8 + file

                # Set the bit
                self.bitboards[index] |= np.uint64(1 << square)
                file += 1

        # Set the side to move
        self.side = WHITE if turn_part =='w' else BLACK

        # Update occupancies
        self.update_occupancies()

    def update_occupancies(self):
        """
        Combine individual bitboards into global occupancy boards.
        """

        # White piece loop
        for piece in range(6):
            self.occupancies[WHITE] |= self.bitboards[piece]

        # Black piece loop
        for piece in range(6):
            self.occupancies[BLACK] |= self.bitboards[piece + 6]

        # All occupancies
        self.occupancies[2] = self.occupancies[WHITE] | self.occupancies[BLACK]

    def get_piece_squares(self, bitboard):
        """
        Extracts all sqaure indices from a bitboard.
        Returns a list of sqaures (0 - 63) where bits are set.
        """
        squares = []

        # Copy of a bitboard
        temp = bitboard

        # While there are still bits set
        while temp:
            least_significant_bit = temp & -temp

            # Convert bits to square index
            # bit_length() gives us the position
            square = least_significant_bit.bit_length() - 1
            squares.append(square)

            temp &= temp - 1

        return squares
    
    def get_piece_at_square(self, square, side):
        """
        Returns the piece present on a given square
        """
        # Bitmask for this specific square
        square_mask = np.uint64(1) << square

        for piece_type in range(6):
            piece_bitboard = self.bitboards[piece_type + (side * 6)]

            if piece_bitboard & square_mask:
                return piece_type
            
        return None

    def get_king_square(self, side):
        """
        Find the square where the king of the given side is located. Returns the square index (0 - 63), -1 if not found.
        """
        king_bitboard = self.bitboards[KING + (side * 6)]

        if king_bitboard == 0:
            return -1 # Shouldn't happen ideally (in chess King is always present on the board)

        return self.get_piece_squares(king_bitboard)[0]

    def encode_move(self, source, target, piece, captured = 0, promotion = 0, flag = 0):
        """
        Encode a move into a 21-bit integer.

        Params:
        - source: source square (0-63)
        - target: target square (0-63)
        - piece: moving piece type
        - captured: captured piece type (0-5. 0 = no capture)
        - promotion: promotion piece type (0-4. 0 = no promotion)
        - flag: special flag (0-7)

        Returns: encoded - move integer
        """
        move = 0

        # Source square bits 0-5
        move |= source
        
        # Target square
        move |= (target << 6)

        # Piece type
        move |= (piece << 12)

        # Captured piece
        move |= (captured << 15)

        # Flags
        move |= (flag << 18)

        return move

    def decode_move(self, move):
        """
        Decode a move integer into its components
        """
        source = move & 0x3F
        target = (move >> 6) & 0x3F
        piece = (move >> 12) & 0x7
        captured = (move >> 15) & 0x7
        flag = (move >> 18) & 0x7

        return {
            'from': source,
            'to': target,
            'piece': piece,
            'captured': captured,
            'flag': flag
        }
    
    def generate_king_moves(self, side):
        """
        Generates all the pseudo-legal moves for the King.
        """
        moves = []

        # Get king bitboard for this side
        king_bitboard = self.bitboards[KING + (side * 6)]
        king_squares = self.get_piece_squares(king_bitboard)
        own_pieces = self.occupancies[side]

        for king_sq in king_squares:
            attack_bitboard = king_attacks_table[king_sq]
            legal_targets = attack_bitboard & ~own_pieces
            target_squares = self.get_piece_squares(legal_targets)

            for target_sq in target_squares:
                captured_piece = self.get_piece_at_square(target_sq, 1 - side)

                if captured_piece is None:
                    captured_piece = 0

                move = self.encode_move(
                    source = king_sq,
                    target = target_sq,
                    piece = KING,
                    captured = captured_piece,
                    promotion = 0,
                    flag = MOVE_FLAG_NORMAL
                )
                moves.append(move)

        return moves
    
    def generate_knight_moves(self, side):
        """
        Generates all the pseudo-legal moves for the knight.
        """
        moves = []

        # Get knight bitboard for this side
        knight_bitboard = self.bitboards[KNIGHT + (side * 6)]
        knight_squares = self.get_piece_squares(knight_bitboard)
        own_pieces = self.occupancies[side]

        for knight_sq in knight_squares:
            attack_bitboard = knight_attacks_table[knight_sq]
            legal_targets = attack_bitboard & ~own_pieces
            target_squares = self.get_piece_squares(legal_targets)

            for target_sq in target_squares:
                captured_piece = self.get_piece_at_square(target_sq, 1 - side)

                if captured_piece is None:
                    captured_piece = 0
                
                move = self.encode_move(
                    source = knight_sq,
                    target = target_sq,
                    piece = KNIGHT,
                    captured = captured_piece,
                    promotion = 0,
                    flag = MOVE_FLAG_NORMAL
                )
                moves.append(move)

        return moves

    def generate_pawn_moves(self, side):
        moves = []

        pawn_bitboard = self.bitboards[PAWN + (side * 6)]
        pawn_squares = self.get_piece_squares(pawn_bitboard)
        occupancies = self.occupancies[2]

        for source_sq in pawn_squares:
            # Determine direction by side
            direction = 8 if side == WHITE else -8
            target_sq = source_sq + direction

            # Check if square in front is empty
            if 0 <= target_sq <= 63 and not(occupancies & (np.uint64(1) << np.uint64(target_sq))):
                # Is it a promotion
                if (side == WHITE and target_sq >= 56) or (side == BLACK and target_sq <= 7):
                    # We have to generate 4 seperate moves for Q, R, B, N
                    for promo in [MOVE_FLAG_PROMOTION_QUEEN, MOVE_FLAG_PROMOTION_ROOK,
                                  MOVE_FLAG_PROMOTION_BISHOP, MOVE_FLAG_PROMOTION_KNIGHT]:
                        moves.append(self.encode_move(source_sq, target_sq, PAWN, flag = promo))
                else:
                    # Normal push
                    moves.append(self.encode_move(source_sq, target_sq, PAWN))

                    # Double pawn push (only if single pawn push was empty)
                    # Check if pawn is on it's starting square
                    is_start_rank = (side == WHITE and source_sq >= 8 and source_sq <= 15) or (side == BLACK and source_sq >= 48 and source_sq <= 55)
                    double_target = source_sq + (direction * 2)

                    if is_start_rank and not (occupancies & (np.uint64(1) << np.uint64(double_target))):
                        moves.append(self.encode_move(source_sq, double_target, PAWN, flag = MOVE_FLAG_DOUBLE_PAWN_PUSH))

            attacks = pawn_attacks_table[side][source_sq]
            opponent_side = 1 - side
            targets = attacks & self.occupancies[opponent_side]
            target_squares = self.get_piece_squares(targets)

            for target_sq in target_squares:
                # Get captured piece type
                captured_piece = self.get_piece_at_square(target_sq, opponent_side)
                
                # Check for promotion capture
                if (side == WHITE and target_sq >= 56) or (side == BLACK and target_sq <= 7):
                    for promo in [MOVE_FLAG_PROMOTION_QUEEN, MOVE_FLAG_PROMOTION_ROOK,
                                  MOVE_FLAG_PROMOTION_BISHOP, MOVE_FLAG_PROMOTION_KNIGHT]:
                        moves.append(self.encode_move(source_sq, target_sq, PAWN, captured = captured_piece, flag = promo))
                else:
                    moves.append(self.encode_move(source_sq, target_sq, PAWN, captured = captured_piece))
            
            # En passant logic
            if self.en_passant_sq != -1:
                # If any of the square we capturing is en passant square
                ep_attacks = attacks & (np.uint64(1) << np.uint64(self.en_passant_sq))
                if ep_attacks:
                    moves.append(self.encode_move(source_sq, self.en_passant_sq, PAWN, captured = PAWN, flag = MOVE_FLAG_EN_PASSANT))
                    
        return moves

    

if __name__ == "__main__":
    # Create the engine
    engine = BoardState()

    # Standard start position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine.parse_fen(start_fen)

    # Test 1: Verification
    # Print the white pawns bitboard
    print_bitboard(engine.bitboards[0])

    # Test 2: All occupied squares
    print_bitboard(engine.occupancies[2])

    # Test 3: King on d4
    print_bitboard(king_attacks_table[D4])

    # Test 4: Knight on e5
    print_bitboard(knight_attacks_table[E5])

    # Test 5: White pawn on b3
    print_bitboard(pawn_attacks_table[WHITE][B3])