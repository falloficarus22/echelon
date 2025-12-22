from constants import *
import numpy as np


def mask_king_attack(square):
    """
    Returns a bitboard of all the king pseudo-moves.
    (Doesn't check for empty square or own pieces yet)
    """
    attacks = np.uint64(0)

    # Rank and files or the current square
    c_rank = square // 8
    c_file = square % 8

    # Loop over steps: -1 (left/down), 0 (stay) and 1 (right/up)\
    for rank_step in [-1, 0, 1]:
        for file_step in [-1, 0, 1]:
            # Skip the current square
            if rank_step == 0 and file_step == 0:
                continue

            # Target rank and file
            t_rank = c_rank + rank_step
            t_file = c_file + file_step

            # Check if the square is within the boundaries of the board
            if 0 <= t_rank <= 7 and 0 <= t_file <= 7:
                # Back to 1D index
                target_sq = t_rank * 8 + t_file

                # Set the bit
                attacks |= np.uint64(1 << target_sq)

    return attacks


def mask_knight_attack(square):
    """
    Returns a bitboard of all pseudo knight-moves
    """
    attacks = np.uint64(0)

    c_rank = square // 8
    c_file = square % 8

    # List of all the knight jumps: rank-change, file-change
    jumps = [(2, -1), (2, 1), (-2, -1), (-2, 1), (1, -2), (1, 2), (-1, -2), (-1, 2)]

    for rank_step, file_step in jumps:
        t_rank = c_rank + rank_step
        t_file = c_file + file_step

        if 0 <= t_rank <= 7 and 0 <= t_file <= 7:
            target_sq = t_rank * 8 + t_file

            attacks |= np.uint64(1 << target_sq)

    return attacks


# Initialize Leaper Attack Tables
# Bitboard of 64 zeros
king_attacks_table = np.zeros(64, dtype=np.uint64)
knight_attacks_table = np.zeros(64, dtype=np.uint64)


def init_leapers():
    """
    Populates the king and knight attack tables.
    Will be called when the engine starts.
    """
    for square in range(64):
        king_attacks_table[square] = mask_king_attack(square)
        knight_attacks_table[square] = mask_knight_attack(square)


# The leaper function is called immediately when attacks.py is used
init_leapers()


def mask_pawn_attack(square, side):
    attacks = np.uint64(0)
    pawn_bitboard = np.uint64(1) << np.uint64(square)

    if side == WHITE:
        # Check if not on file A before capturing to the left
        if not (pawn_bitboard & FILE_A):
            attacks |= pawn_bitboard << np.uint64(7)
        # Check if not on file H before capturing to the right
        if not (pawn_bitboard & FILE_H):
            attacks |= pawn_bitboard << np.uint64(9)
    else:
        # Check if not on file A before capturing to the left
        if not (pawn_bitboard & FILE_A):
            attacks |= pawn_bitboard >> np.uint64(7)
        # Check if not on File H before capturing to the right
        if not (pawn_bitboard & FILE_H):
            attacks |= pawn_bitboard >> np.uint64(9)

    return attacks


pawn_attacks_table = np.zeros((2, 64), dtype=np.uint64)


def init_pawns():
    """
    Populates the pawn attack table.
    """
    for square in range(64):
        pawn_attacks_table[WHITE][square] = mask_pawn_attack(square, WHITE)
        pawn_attacks_table[BLACK][square] = mask_pawn_attack(square, BLACK)


init_pawns()


def generate_ray_attacks(square, occupancy, direction):
    """
    Generates a bitboard of attacks of sliding pieces through a single ray,
    stopping at the first blocker.

    This function is primarily used to pre-calculate the Magic Lookup Tables.
    """

    attacks = np.uint64(0)
    curr_sq = square

    # Moving one step at a time until the boundary of the board
    while True:
        curr_sq += direction

        # Check if the new square is outside the 8x8 boundary
        if not (0 <= curr_sq <= 63):
            break

        # Check for wrap-around
        # For Rooks: Check file/rank consistency
        if direction == EAST and (curr_sq % 8 == 0):
            break
        if direction == WEST and (curr_sq % 8 == 7):
            break

        # For Bishops: Check if the distance to the edge is correct

        # Set the attack bit
        attacks |= np.uint64(1) << curr_sq

        # Check for a blocker
        if occupancy & (np.uint64(1) << curr_sq):
            break

    return attacks


def generate_bishop_attacks(square, occupancy):
    """
    Generates all the pseudo-legal moves for the bishop.
    """

    attacks = np.uint64(0)

    # Diagonal directions
    directions = [NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST]

    for direction in directions:
        attacks |= generate_ray_attacks(square, occupancy, direction)

    return attacks


def generate_rook_attacks(square, occupancy):
    """
    Generates all the pseudo-legal moves for the rook.
    """

    attacks = np.uint64(0)

    # Directions for rook
    directions = [EAST, WEST, NORTH, SOUTH]

    for direction in directions:
        attacks |= generate_ray_attacks(square, occupancy, direction)

    return attacks


def generate_rook_mask(square):
    """
    Generates the mask of all relevant inner squares for a Rook.
    (Excluding the edges of the board and the home square)
    """

    attacks = np.uint64(0)

    directions = [EAST, WEST, NORTH, SOUTH]

    for direction in directions:
        curr_sq = square

        # Move one step away from the starting square
        curr_sq += direction

        while 0 <= curr_sq <= 63:
            attacks |= np.uint64(1) << curr_sq
            curr_sq += direction

            # If the next step is an edge, break
            if not (0 <= curr_sq <= 63):
                break
            if direction == EAST and (curr_sq % 8 == 0):
                break
            if direction == WEST and (curr_sq % 8 == 7):
                break
            if direction == NORTH and (curr_sq // 8 == 7):
                break
            if direction == SOUTH and (curr_sq // 8 == 0):
                break

    return attacks


def generate_bishop_mask(square):
    """
    Generates the mask of all relevant inner squares for a Bishop.
    (Excluding the edges of the board and the home square)
    """
    # Similar to rook masks but different directions and
    # edge detection logic
    attacks = np.uint64(0)
    directions = [NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST]

    for direction in directions:
        curr_sq = square
        curr_sq += direction

        while 0 <= curr_sq <= 63:
            attacks |= np.uint64(1) << curr_sq
            curr_sq += direction

            # Boundary check if ray hits the edge of the board
            if not (0 <= curr_sq <= 63):
                break
            if (
                (curr_sq % 8 == 0)
                or (curr_sq % 8 == 7)
                or (curr_sq // 8 == 0)
                or (curr_sq // 8 == 7)
            ):
                break

    return attacks


def set_occupancy(index, bits_in_mask, attacks_mask):
    """
    Returns a specific occupancy configuration from a mask.
    index: which configurations to generate (0 to 2^bits_in_mask)
    """
    occupancy = np.uint64(0)
    mask_copy = attacks_mask

    for i in range(bits_in_mask):
        # Find the square of the i-th set bit in the mask
        i, square = pop_bit(mask_copy)

        # Clear that bit from the mask copy
        attacks_mask &= attacks_mask - np.uint64(1)

        # If the i-th bit of our index is set, set this square in our occupancy
        if index & (1 << i):
            occupancy |= np.uint64(1) << np.uint64(square)

    return occupancy


def get_lsb_index(bitboard):
    """
    Helper function to get the index of the least significant bit.
    """
    if bitboard == 0:
        return -1

    lsb = bitboard & -bitboard
    return int(np.log2(lsb))


def pop_bit(bitboard):
    """
    Remove and return the least significant bit.
    """
    if bitboard == 0:
        return bitboard, -1

    index = get_lsb_index(bitboard)
    return bitboard & (bitboard - np.uint64(1)), index


def count_bits(bitboard):
    """
    Count the number of set bits in bitboard
    """
    count = 0
    while bitboard:
        bitboard &= bitboard - np.uint64(1)
        count += 1

    return count
