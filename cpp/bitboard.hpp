#pragma once

#include "constants.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace Bitboard {

// Get bit from bitboard
inline bool get_bit(U64 bb, int square) {
    return (bb >> square) & 1ULL;
}

// Set bit on bitboard
inline void set_bit(U64 &bb, int square) {
    bb |= (1ULL << square);
}

// Pop bit from square
inline void pop_bit(U64 &bb, int square) {
    bb &= ~(1ULL << square);
}

// Count set bits (Hardware Popcount)
inline int count_bits(U64 bb) {
#ifdef _MSC_VER
    return (int)__popcnt64(bb);
#else
    return __builtin_popcountll(bb);
#endif
}

// Get Least Significant Bit index
inline int get_lsb_index(U64 bb) {
    if (bb == 0) return -1;
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward64(&index, bb);
    return (int)index;
#else
    return __builtin_ctzll(bb);
#endif
}

// Strip LSB and return its index
inline int pop_lsb(U64 &bb) {
    int index = get_lsb_index(bb);
    bb &= (bb - 1);
    return index;
}

} // namespace Bitboard
