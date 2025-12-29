import numpy as np
from constants import *
from attacks import (
    generate_rook_attacks,
    generate_bishop_attacks,
    generate_rook_mask,
    generate_bishop_mask,
    set_occupancy,
    count_bits,
)
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Pre computed magic numbers for rooks
ROOK_MAGICS = [
    0x0080001020400080,
    0x0040001000200040,
    0x0080081000200080,
    0x0080040800100080,
    0x0080020400080080,
    0x0080010200040080,
    0x0080008001000200,
    0x0080002040800100,
    0x0000800020400080,
    0x0000400020005000,
    0x0000801000200080,
    0x0000800800100080,
    0x0000800400080080,
    0x0000800200040080,
    0x0000800100020080,
    0x0000800040800100,
    0x0000208000400080,
    0x0000404000201000,
    0x0000808010000800,
    0x0000808008000400,
    0x0000808004000200,
    0x0000808002000100,
    0x0000808001000100,
    0x0000408000800100,
    0x0000204000808000,
    0x0000200040008080,
    0x0000100080004080,
    0x0000080080002080,
    0x0000040080001080,
    0x0000020080000880,
    0x0000010080000480,
    0x0000008080000280,
    0x0000804000800020,
    0x0000402000401000,
    0x0000801000200080,
    0x0000800800100080,
    0x0000800400080080,
    0x0000800200040080,
    0x0000800100020080,
    0x0000800040800100,
    0x0000208000400080,
    0x0000404000201000,
    0x0000808010000800,
    0x0000808008000400,
    0x0000808004000200,
    0x0000808002000100,
    0x0000808001000100,
    0x0000408000800100,
    0x0000204000808000,
    0x0000200040008080,
    0x0000100080004080,
    0x0000080080002080,
    0x0000040080001080,
    0x0000020080000880,
    0x0000010080000480,
    0x0000008080000280,
    0x0000800020400080,
    0x0000400020005000,
    0x0000801000200080,
    0x0000800800100080,
    0x0000800400080080,
    0x0000800200040080,
    0x0000800100020080,
    0x0000800040800100,
]

# Pre computed magic numbers for bishops
BISHOP_MAGICS = [
    0x0002020202020200,
    0x0002020202020000,
    0x0004010202000000,
    0x0004040080000000,
    0x0001104000000000,
    0x0000821040000000,
    0x0000410410400000,
    0x0000104104104000,
    0x0000040404040400,
    0x0000020202020200,
    0x0000040102020000,
    0x0000040400800000,
    0x0000011040000000,
    0x0000008210400000,
    0x0000004104104000,
    0x0000002082082000,
    0x0004000808080800,
    0x0002000404040400,
    0x0001000202020200,
    0x0000800802004000,
    0x0000800400A00000,
    0x0000200100884000,
    0x0000400082082000,
    0x0000200041041000,
    0x0002080010101000,
    0x0001040008080800,
    0x0000208004010400,
    0x0000404004010200,
    0x0000840000802000,
    0x0000404002011000,
    0x0000808001041000,
    0x0000404000820800,
    0x0001041000202000,
    0x0000820800101000,
    0x0000104400080800,
    0x0000020080080080,
    0x0000404040040100,
    0x0000808100020100,
    0x0001010100020800,
    0x0000808080010400,
    0x0000820820004000,
    0x0000410410002000,
    0x0000082088001000,
    0x0000002011000800,
    0x0000080100400400,
    0x0001010101000200,
    0x0002020202000400,
    0x0001010101000200,
    0x0000410410400000,
    0x0000208208200000,
    0x0000002084100000,
    0x0000000020880000,
    0x0000001002020000,
    0x0000040408020000,
    0x0004040404040000,
    0x0002020202020000,
    0x0000104104104000,
    0x0000002082082000,
    0x0000000020841000,
    0x0000000000208800,
    0x0000000010020200,
    0x0000000404080200,
    0x0000040404040400,
    0x0002020202020200,
]

# Bit counts for each square (number of relevant bits in the mask)
ROOK_RELEVANT_BITS = [
    12,
    11,
    11,
    11,
    11,
    11,
    11,
    12,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    12,
    11,
    11,
    11,
    11,
    11,
    11,
    12,
]

BISHOP_RELEVANT_BITS = [
    6,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    7,
    7,
    7,
    7,
    5,
    5,
    5,
    5,
    7,
    9,
    9,
    7,
    5,
    5,
    5,
    5,
    7,
    9,
    9,
    7,
    5,
    5,
    5,
    5,
    7,
    7,
    7,
    7,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
]


def init_magic_tables():
    """
    Initialize the magic bitboard lookup tables for bishops and rooks.
    Pre computes all the possible attack patterns for every square and occupancy.
    Called once at the engine startup
    """
    print("Initializing magic bitboard tables...")

    # Initialize rook attacks
    for square in range(64):
        mask = generate_rook_mask(square)
        relevant_bits = ROOK_RELEVANT_BITS[square]

        # Generate all the possible occupancy patterns
        occupancy_variations = 1 << relevant_bits

        for index in range(occupancy_variations):
            occupancy = set_occupancy(index, relevant_bits, mask)

            # Calculate the magic index for this occupancy
            magic_index = (occupancy * ROOK_MAGICS[square]) >> (64 - relevant_bits)

            # Store the pre-computed attacks for this square
            rook_attacks[square][magic_index] = generate_rook_attacks(square, occupancy)

    # Initialize bishop attacks
    for square in range(64):
        mask = generate_bishop_mask(square)
        relevant_bits = BISHOP_RELEVANT_BITS[square]
        occupancy_variations = 1 << relevant_bits

        for index in range(occupancy_variations):
            occupancy = set_occupancy(index, relevant_bits, mask)

            magic_index = (occupancy * BISHOP_MAGICS[square]) >> (64 - relevant_bits)
            bishop_attacks[square][magic_index] = generate_bishop_attacks(square, occupancy)

    print("Magic tables initialized successfully.")


def get_rook_attacks(square, occupancy):
    """
    Gets rook attacks for a given square and occupancy.
    """
    # Mask only the relevant squares for this rook
    relevant_occupancy = occupancy & generate_rook_mask(square)

    # Calculate the magic index using magic multiplication
    magic_index = (relevant_occupancy * ROOK_MAGICS[square]) >> (64 - ROOK_RELEVANT_BITS[square])

    return rook_attacks[square][magic_index]


def get_bishop_attacks(square, occupancy):
    """
    Get bishop attacks for a give square and occupancy
    """
    relevant_occupancy = occupancy & generate_bishop_mask(square)
    magic_index = (relevant_occupancy * BISHOP_MAGICS[square]) >> (
        64 - BISHOP_RELEVANT_BITS[square]
    )

    return bishop_attacks[square][magic_index]


def get_queen_attacks(square, occupancy):
    """
    Get queen attacks (combination of rooks and bishops)
    """
    return get_rook_attacks(square, occupancy) | get_bishop_attacks(square, occupancy)


# Initialize magic tables when the module is imported
init_magic_tables()

# Test the magic bitboards
if __name__ == "__main__":
    print("\nTesting Magic Bitboards")

    # Test rook on d4 with some occupancy
    test_occupancy = np.uint64(0x0000001008100000)
    rook_attacks_d4 = get_rook_attacks(D4, test_occupancy)

    print("\nRook on D4 with occupancy:")
    print_bitboard(test_occupancy)
    print("Rook attacks:")
    print_bitboard(rook_attacks_d4)

    # Test bishop on e4
    bishop_attacks_e4 = get_bishop_attacks(E4, test_occupancy)
    print("Bishop on E4 with same occupancy:")
    print_bitboard(bishop_attacks_e4)

    # Test queen
    queen_attacks_e4 = get_queen_attacks(E4, test_occupancy)
    print("Queen on E4 (rook + bishop):")
    print_bitboard(queen_attacks_e4)
