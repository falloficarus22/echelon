import numpy as np

# Colors and Pieces

WHITE = 0
BLACK = 1

# Piece Types

PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5

# Printing the board to console:
# Indices 0-5 are black pieces and 6-11 are white pieces

PIECE_SYMBOLS = "pnbrqkPNBRQK"

# Square Mapping using the Little Endian Rank-File Mapping

MAPPING = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8
] = range(64)

# Move Directions

# Horizontal-Vertical
NORTH = 8
SOUTH = -8
EAST = 1
WEST = -1

# Diagonal
NORTH_EAST = 9
SOUTH_EAST = -7
NORTH_WEST = 7
SOUTH_WEST = -9

# Move Flags
MOVE_FLAG_NORMAL = 0
MOVE_FLAG_PROMOTION_KNIGHT = 1
MOVE_FLAG_PROMOTION_BISHOP = 2
MOVE_FLAG_PROMOTION_ROOK = 3
MOVE_FLAG_PROMOTION_QUEEN = 4
MOVE_FLAG_DOUBLE_PAWN_PUSH = 5
MOVE_FLAG_EN_PASSANT = 6
MOVE_FLAG_CASTLING = 7

# Bitboard mask
FILE_A = np.uint64(0x0101010101010101)
FILE_B = np.uint64(0x0202020202020202)
FILE_C = np.uint64(0x0404040404040404)
FILE_D = np.uint64(0x0808080808080808)
FILE_E = np.uint64(0x1010101010101010)
FILE_F = np.uint64(0x2020202020202020)
FILE_G = np.uint64(0x4040404040404040)
FILE_H = np.uint64(0x8080808080808080)

RANK_1 = np.uint64(0x00000000000000FF)
RANK_2 = np.uint64(0x000000000000FF00)
RANK_3 = np.uint64(0x0000000000FF0000)
RANK_4 = np.uint64(0x00000000FF000000)
RANK_5 = np.uint64(0x000000FF00000000)
RANK_6 = np.uint64(0x0000FF0000000000)
RANK_7 = np.uint64(0x00FF000000000000)
RANK_8 = np.uint64(0xFF00000000000000)

def print_bitboard(bitboard):
    """
    Prints a bitboard as an 8x8 grid
    """
    print()
    for rank in range(8):
        # Print rank label
        print(f" {8 - rank}", end = "")

        for file in range(8):
            # Map 2D coordinates to 1D index
            square = rank * 8 + file
            square_idx = (7 - rank) * 8 + file

            # Check if the bit is set to 1 and not 0
            if bitboard & (1 << square_idx):
                print(" 1", end = "")
            else:
                print(" .", end = "")

        # New line after every rank
        print()

    print("\n   ", end = "")
    for file in range(8):
        print(f"{chr(ord('a') + file)} ", end = "")
    print("\n")
    print()

