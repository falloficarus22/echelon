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
    jumps = [
        (2, -1), (2, 1), (-2, -1), (-2, 1),
        (1, -2), (1, 2), (-1, -2), (-1, 2)
    ]

    for rank_step, file_step in jumps:
        t_rank = c_rank + rank_step
        t_file = c_file + file_step

        if 0 <= t_rank <= 7 and 0 <= t_file <= 7:
            target_sq = t_rank * 8 + t_file

            attacks |= np.uint64(1 << target_sq)

    return attacks

# Initialize Leaper Attack Tables
# Bitboard of 64 zeros
king_attacks_table = np.zeros(64, dtype = np.uint64)
knight_attacks_table = np.zeros(64, dtype = np.uint64)

def init_leapers():
    """
    Populates the king and knight attack tables
    Will be called when the engine starts
    """
    for square in range(64):
        king_attacks_table[square] = mask_king_attack(square)
        knight_attacks_table[square] = mask_knight_attack(square)

# The leaper function is called immediately when it is called
init_leapers()