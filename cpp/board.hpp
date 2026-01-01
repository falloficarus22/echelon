#pragma once

#include "constants.hpp"
#include <string>
#include <vector>

// Move structure (Source and target squares + flags)
struct Move {
    int from;
    int to;
    MoveFlag flag;
    
    // Convert to a simple integer for efficient storage
    uint32_t encode() const {
        return (from) | (to << 6) | (flag << 12);
    }
};

#include "bitboard.hpp"

// History for unmaking moves
struct History {
    int en_passant_sq;
    int castle_rights;
    int halfmove_clock;
    int captured_piece;
};

// Move list for move generation
struct MoveList {
    Move moves[256];
    int count = 0;

    void add(Move move) {
        moves[count++] = move;
    }
};

class Board {
public:
    U64 bitboards[12];
    U64 occupancies[3];
    Color side;
    int en_passant_sq;
    int castle_rights;
    int halfmove_clock;
    std::vector<U64> position_history;

    Board();
    
    // Essential methods
    void parse_fen(const std::string &fen);
    void update_occupancies();
    void update_occupancies_fixed();

    U64 get_position_hash() const;
    bool is_threefold_repetition() const;
    bool is_fifty_move_rule() const;
    bool is_draw() const;
    
    // Move logic
    History make_move(Move move);
    void unmake_move(Move move, const History &hist);
    
    // Move generation
    void generate_moves(MoveList &list) const;
    void generate_pawn_moves(MoveList &list) const;
    void generate_leaper_moves(MoveList &list) const;
    void generate_sliding_moves(MoveList &list) const;
    void generate_castling_moves(MoveList &list) const;

    // Tensorization for neural network
    std::vector<float> tensorize() const {
        std::vector<float> data(13 * 8 * 8, 0.0f);
        
        for (int p_idx = 0; p_idx < 12; p_idx++) {
            U64 bb = bitboards[p_idx];
            while (bb) {
                int sq = Bitboard::pop_lsb(bb);
                // Map to [plane, rank, file]
                int rank = sq / 8;
                int file = sq % 8;
                data[p_idx * 64 + rank * 8 + file] = 1.0f;
            }
        }
        
        // 13th plane: Side to move
        if (side == WHITE) {
            for (int i = 0; i < 64; i++) {
                data[12 * 64 + i] = 1.0f;
            }
        }
        
        return data;
    }

    // Attack detection
    bool is_square_attacked(int square, Color side_attacking) const;
    bool is_in_check(int c = -1) const {
        Color side_to_check = (c == -1) ? side : (Color)c;
        // Find king square
        int king_sq = Bitboard::get_lsb_index(bitboards[side_to_check == WHITE ? W_KING : B_KING]);
        if (king_sq == -1) return false;
        return is_square_attacked(king_sq, side_to_check == WHITE ? BLACK : WHITE);
    }
    
    // Evaluation (greedy baseline)
    int evaluate() const;

private:
    // Internal piece tracking
    int get_piece_at(int square) const;
};
