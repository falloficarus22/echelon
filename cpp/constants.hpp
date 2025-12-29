#pragma once

#include <cstdint>

typedef uint64_t U64;

// Colors
enum Color {
    WHITE, BLACK, BOTH
};

// Piece Types
enum PieceType {
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
};

// Global Piece Indices (Matches Python 0-11)
enum Piece {
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING
};

// Squares
enum Square {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8, NO_SQ
};

// Directions
enum Direction {
    NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1,
    NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9
};

// File and Rank Masks (Essential for wrap-around checks)
const U64 FILE_A = 0x0101010101010101ULL;
const U64 FILE_B = 0x0202020202020202ULL;
const U64 FILE_G = 0x4040404040404040ULL;
const U64 FILE_H = 0x8080808080808080ULL;
const U64 RANK_1 = 0x00000000000000FFULL;
const U64 RANK_2 = 0x000000000000FF00ULL;
const U64 RANK_7 = 0x00FF000000000000ULL;
const U64 RANK_8 = 0xFF00000000000000ULL;
const U64 RANK_4 = 0x00000000FF000000ULL; // useful for double push
const U64 RANK_5 = 0x000000FF00000000ULL;

// Move Flags
enum MoveFlag {
    MOVE_FLAG_NORMAL = 0,
    MOVE_FLAG_PROMOTION_KNIGHT = 1,
    MOVE_FLAG_PROMOTION_BISHOP = 2,
    MOVE_FLAG_PROMOTION_ROOK = 3,
    MOVE_FLAG_PROMOTION_QUEEN = 4,
    MOVE_FLAG_DOUBLE_PAWN_PUSH = 5,
    MOVE_FLAG_EN_PASSANT = 6,
    MOVE_FLAG_CASTLING = 7
};
