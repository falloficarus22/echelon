#pragma once

#include "constants.hpp"

namespace Magic {

// Precomputed tables for sliding pieces
extern U64 rook_table[64][4096];
extern U64 bishop_table[64][512];

// Relevant bits for masks
extern const int rook_bits[64];
extern const int bishop_bits[64];

// Magic numbers
extern const U64 rook_magics[64];
extern const U64 bishop_magics[64];

// Masks
extern U64 rook_masks[64];
extern U64 bishop_masks[64];

// Initialization
void init_all();

// Inline attack lookups
inline U64 get_rook_attacks(int square, U64 occupancy) {
    occupancy &= rook_masks[square];
    occupancy *= rook_magics[square];
    return rook_table[square][occupancy >> (64 - rook_bits[square])];
}

inline U64 get_bishop_attacks(int square, U64 occupancy) {
    occupancy &= bishop_masks[square];
    occupancy *= bishop_magics[square];
    return bishop_table[square][occupancy >> (64 - bishop_bits[square])];
}

inline U64 get_queen_attacks(int square, U64 occupancy) {
    return get_rook_attacks(square, occupancy) | get_bishop_attacks(square, occupancy);
}

} // namespace Magic
