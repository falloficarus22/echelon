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

