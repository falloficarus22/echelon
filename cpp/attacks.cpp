#include "attacks.hpp"
#include "bitboard.hpp"
#include <cmath>
#include <iostream>

namespace Attacks {

U64 pawn_attacks[2][64];
U64 knight_attacks[64];
U64 king_attacks[64];

U64 mask_pawn_attacks(int square, int color) {
    U64 attacks = 0ULL;
    U64 bit = (1ULL << square);

    if (color == WHITE) {
        if (!(bit & FILE_A)) attacks |= (bit << 7);
        if (!(bit & FILE_H)) attacks |= (bit << 9);
    } else {
        if (!(bit & FILE_A)) attacks |= (bit >> 9);
        if (!(bit & FILE_H)) attacks |= (bit >> 7);
    }
    return attacks;
}

U64 mask_knight_attacks(int square) {
    U64 attacks = 0ULL;
    U64 bit = (1ULL << square);
    
    // Knight jumps
    if (!(bit & (FILE_A | FILE_B))) {
        attacks |= (bit << 6);  // up 1, left 2
        attacks |= (bit >> 10); // down 1, left 2
    }
    if (!(bit & FILE_A)) {
        attacks |= (bit << 15); // up 2, left 1
        attacks |= (bit >> 17); // down 2, left 1
    }
    if (!(bit & FILE_H)) {
        attacks |= (bit << 17); // up 2, right 1
        attacks |= (bit >> 15); // down 2, right 1
    }
    if (!(bit & (FILE_G | FILE_H))) {
        attacks |= (bit << 10); // up 1, right 2
        attacks |= (bit >> 6);  // down 1, right 2
    }
    return attacks;
}

U64 mask_king_attacks(int square) {
    U64 attacks = 0ULL;
    U64 bit = (1ULL << square);
    
    if (!(bit & FILE_A)) {
        attacks |= (bit << 7);
        attacks |= (bit >> 1);
        attacks |= (bit >> 9);
    }
    if (!(bit & FILE_H)) {
        attacks |= (bit << 9);
        attacks |= (bit << 1);
        attacks |= (bit >> 7);
    }
    attacks |= (bit << 8);
    attacks |= (bit >> 8);
    
    return attacks;
}

U64 generate_ray_attacks(int square, U64 occupancy, Direction dir) {
    U64 attacks = 0ULL;
    int curr_sq = square;

    while (true) {
        int prev_sq = curr_sq;
        curr_sq += dir;

        if (curr_sq < 0 || curr_sq > 63) break;

        // Wrap-around check
        int prev_file = prev_sq % 8;
        int curr_file = curr_sq % 8;

        if (dir == EAST || dir == WEST || dir == NORTH_EAST || dir == NORTH_WEST || dir == SOUTH_EAST || dir == SOUTH_WEST) {
            if (std::abs(curr_file - prev_file) > 1) break;
        }
        
        if ((dir == EAST || dir == WEST) && (curr_file == prev_file)) break;

        attacks |= (1ULL << curr_sq);
        if (occupancy & (1ULL << curr_sq)) break;
    }
    return attacks;
}

void init_all() {
    for (int sq = 0; sq < 64; ++sq) {
        pawn_attacks[WHITE][sq] = mask_pawn_attacks(sq, WHITE);
        pawn_attacks[BLACK][sq] = mask_pawn_attacks(sq, BLACK);
        knight_attacks[sq] = mask_knight_attacks(sq);
        king_attacks[sq] = mask_king_attacks(sq);
    }
}

} // namespace Attacks
