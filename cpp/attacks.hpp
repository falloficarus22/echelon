#pragma once

#include "constants.hpp"
#include <vector>

namespace Attacks {

// Lookups for leaper pieces
extern U64 pawn_attacks[2][64];
extern U64 knight_attacks[64];
extern U64 king_attacks[64];

// Initialization
void init_all();

// Generation helpers (used during init)
U64 mask_pawn_attacks(int square, int color);
U64 mask_knight_attacks(int square);
U64 mask_king_attacks(int square);

// Sliding move generation
U64 generate_ray_attacks(int square, U64 occupancy, Direction dir);

} // namespace Attacks
