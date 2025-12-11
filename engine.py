import numpy as np
from constants import *
from attacks import king_attacks_table, knight_attacks_table

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

