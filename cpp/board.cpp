#include "board.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"
#include "magic.hpp"
#include <iostream>

Board::Board() {
    for (int i = 0; i < 12; i++) bitboards[i] = 0ULL;
    for (int i = 0; i < 3; i++) occupancies[i] = 0ULL;
    side = WHITE;
    en_passant_sq = -1;
    castle_rights = 0b1111; // All rights initially
    halfmove_clock = 0;
}

void Board::update_occupancies() {
    occupancies[WHITE] = 0;
    occupancies[BLACK] = 0;
    for (int i = 0; i < 6; i++) occupancies[WHITE] |= bitboards[i];
    for (int i = 6; i < 12; i++) occupancies[BLACK] |= bitboards[i];
    occupancies[BOTH] = occupancies[WHITE] | occupancies[BLACK];
}

void Board::parse_fen(const std::string &fen) {
    for (int i = 0; i < 12; i++) bitboards[i] = 0ULL;
    
    int rank = 7, file = 0;
    size_t i = 0;
    for (; i < fen.length() && fen[i] != ' '; i++) {
        char c = fen[i];
        if (c == '/') { rank--; file = 0; }
        else if (isdigit(c)) { file += (c - '0'); }
        else {
            int sq = rank * 8 + file;
            if (c == 'P') Bitboard::set_bit(bitboards[W_PAWN], sq);
            else if (c == 'N') Bitboard::set_bit(bitboards[W_KNIGHT], sq);
            else if (c == 'B') Bitboard::set_bit(bitboards[W_BISHOP], sq);
            else if (c == 'R') Bitboard::set_bit(bitboards[W_ROOK], sq);
            else if (c == 'Q') Bitboard::set_bit(bitboards[W_QUEEN], sq);
            else if (c == 'K') Bitboard::set_bit(bitboards[W_KING], sq);
            else if (c == 'p') Bitboard::set_bit(bitboards[B_PAWN], sq);
            else if (c == 'n') Bitboard::set_bit(bitboards[B_KNIGHT], sq);
            else if (c == 'b') Bitboard::set_bit(bitboards[B_BISHOP], sq);
            else if (c == 'r') Bitboard::set_bit(bitboards[B_ROOK], sq);
            else if (c == 'q') Bitboard::set_bit(bitboards[B_QUEEN], sq);
            else if (c == 'k') Bitboard::set_bit(bitboards[B_KING], sq);
            file++;
        }
    }
    
    // Parse turn
    while (i < fen.length() && fen[i] == ' ') i++;
    if (i < fen.length()) {
        side = (fen[i] == 'w') ? WHITE : BLACK;
        i++;
    }

    // Update occupancies
    update_occupancies();
}

bool Board::is_square_attacked(int square, Color attacking_side) const {
    U64 occ = occupancies[BOTH];
    
    // Pawn attacks
    if (attacking_side == WHITE) {
        if (Attacks::pawn_attacks[BLACK][square] & bitboards[W_PAWN]) return true;
    } else {
        if (Attacks::pawn_attacks[WHITE][square] & bitboards[B_PAWN]) return true;
    }
    
    // Knight attacks
    if (Attacks::knight_attacks[square] & bitboards[attacking_side == WHITE ? W_KNIGHT : B_KNIGHT]) return true;
    
    // King attacks
    if (Attacks::king_attacks[square] & bitboards[attacking_side == WHITE ? W_KING : B_KING]) return true;
    
    // Sliding attacks (Magic)
    if (Magic::get_bishop_attacks(square, occ) & (bitboards[attacking_side == WHITE ? W_BISHOP : B_BISHOP] | bitboards[attacking_side == WHITE ? W_QUEEN : B_QUEEN])) return true;
    if (Magic::get_rook_attacks(square, occ) & (bitboards[attacking_side == WHITE ? W_ROOK : B_ROOK] | bitboards[attacking_side == WHITE ? W_QUEEN : B_QUEEN])) return true;
    
    return false;
}

int Board::evaluate() const {
    static const int values[] = {100, 320, 330, 500, 900, 20000};
    int score = 0;
    for (int i = 0; i < 6; i++) {
        score += Bitboard::count_bits(bitboards[i]) * values[i];
        score -= Bitboard::count_bits(bitboards[i+6]) * values[i];
    }
    return (side == WHITE) ? score : -score;
}

int Board::get_piece_at(int square) const {
    for (int i = 0; i < 12; i++) {
        if (Bitboard::get_bit(bitboards[i], square)) return i;
    }
    return -1;
}

History Board::make_move(Move move) {
    History hist;
    hist.en_passant_sq = en_passant_sq;
    hist.castle_rights = castle_rights;
    hist.halfmove_clock = halfmove_clock;
    
    int piece = get_piece_at(move.from);
    int captured = get_piece_at(move.to);
    hist.captured_piece = (captured == -1) ? -1 : (captured % 6);

    // Default: clear en passant
    en_passant_sq = -1;
    
    // Move piece on bitboards
    Bitboard::pop_bit(bitboards[piece], move.from);
    
    if (move.flag == MOVE_FLAG_EN_PASSANT) {
        int cap_sq = (side == WHITE) ? move.to - 8 : move.to + 8;
        Bitboard::pop_bit(bitboards[side == WHITE ? B_PAWN : W_PAWN], cap_sq);
    } else if (captured != -1) {
        Bitboard::pop_bit(bitboards[captured], move.to);
    }

    if (move.flag >= MOVE_FLAG_PROMOTION_KNIGHT && move.flag <= MOVE_FLAG_PROMOTION_QUEEN) {
        int promo_piece = -1;
        if (move.flag == MOVE_FLAG_PROMOTION_KNIGHT) promo_piece = KNIGHT;
        else if (move.flag == MOVE_FLAG_PROMOTION_BISHOP) promo_piece = BISHOP;
        else if (move.flag == MOVE_FLAG_PROMOTION_ROOK) promo_piece = ROOK;
        else if (move.flag == MOVE_FLAG_PROMOTION_QUEEN) promo_piece = QUEEN;
        Bitboard::set_bit(bitboards[promo_piece + (side * 6)], move.to);
    } else {
        Bitboard::set_bit(bitboards[piece], move.to);
    }

    // Special cases
    if (move.flag == MOVE_FLAG_DOUBLE_PAWN_PUSH) {
        en_passant_sq = (side == WHITE) ? move.from + 8 : move.from - 8;
    } else if (move.flag == MOVE_FLAG_CASTLING) {
        // Move Rook too
        int r_from, r_to;
        if (move.to > move.from) { // Kingside
            r_from = (side == WHITE) ? H1 : H8;
            r_to = (side == WHITE) ? F1 : F8;
        } else { // Queenside
            r_from = (side == WHITE) ? A1 : A8;
            r_to = (side == WHITE) ? D1 : D8;
        }
        Bitboard::pop_bit(bitboards[ROOK + (side * 6)], r_from);
        Bitboard::set_bit(bitboards[ROOK + (side * 6)], r_to);
    }

    // Update Castle Rights
    static const int rights_mask[64] = {
        13, 15, 15, 15, 12, 15, 15, 14,
        15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        7, 15, 15, 15, 3, 15, 15, 11
    };
    castle_rights &= rights_mask[move.from];
    castle_rights &= rights_mask[move.to];

    // Clock
    if ((piece % 6) == PAWN || captured != -1) halfmove_clock = 0;
    else halfmove_clock++;

    side = (side == WHITE) ? BLACK : WHITE;
    update_occupancies();
    return hist;
}

void Board::unmake_move(Move move, const History &hist) {
    side = (side == WHITE) ? BLACK : WHITE;
    
    int piece = get_piece_at(move.to);
    
    // Remove from target
    Bitboard::pop_bit(bitboards[piece], move.to);
    
    // Restore piece to source
    if (move.flag >= MOVE_FLAG_PROMOTION_KNIGHT && move.flag <= MOVE_FLAG_PROMOTION_QUEEN) {
        Bitboard::set_bit(bitboards[PAWN + (side * 6)], move.from);
    } else {
        Bitboard::set_bit(bitboards[piece], move.from);
    }

    // Restore capture
    if (move.flag == MOVE_FLAG_EN_PASSANT) {
        int cap_sq = (side == WHITE) ? move.to - 8 : move.to + 8;
        Bitboard::set_bit(bitboards[side == WHITE ? B_PAWN : W_PAWN], cap_sq);
    } else if (hist.captured_piece != -1) {
        Bitboard::set_bit(bitboards[hist.captured_piece + ((side == WHITE ? BLACK : WHITE) * 6)], move.to);
    }

    // Restore Castle Rook
    if (move.flag == MOVE_FLAG_CASTLING) {
        int r_from, r_to;
        if (move.to > move.from) { // Kingside
            r_from = (side == WHITE) ? H1 : H8;
            r_to = (side == WHITE) ? F1 : F8;
        } else { // Queenside
            r_from = (side == WHITE) ? A1 : A8;
            r_to = (side == WHITE) ? D1 : D8;
        }
        Bitboard::pop_bit(bitboards[ROOK + (side * 6)], r_to);
        Bitboard::set_bit(bitboards[ROOK + (side * 6)], r_from);
    }

    en_passant_sq = hist.en_passant_sq;
    castle_rights = hist.castle_rights;
    halfmove_clock = hist.halfmove_clock;
    
    update_occupancies();
}

void Board::generate_moves(MoveList &list) const {
    generate_pawn_moves(list);
    generate_leaper_moves(list);
    generate_sliding_moves(list);
    generate_castling_moves(list);
}

void Board::generate_pawn_moves(MoveList &list) const {
    U64 pawns = bitboards[side == WHITE ? W_PAWN : B_PAWN];
    
    while (pawns) {
        int from = Bitboard::pop_lsb(pawns);
        int to;

        // Quiet moves
        to = (side == WHITE) ? from + 8 : from - 8;
        if (to >= 0 && to <= 63 && !Bitboard::get_bit(occupancies[BOTH], to)) {
            // Promotion
            if ((side == WHITE && to / 8 == 7) || (side == BLACK && to / 8 == 0)) {
                list.add({from, to, MOVE_FLAG_PROMOTION_QUEEN});
                list.add({from, to, MOVE_FLAG_PROMOTION_ROOK});
                list.add({from, to, MOVE_FLAG_PROMOTION_BISHOP});
                list.add({from, to, MOVE_FLAG_PROMOTION_KNIGHT});
            } else {
                list.add({from, to, MOVE_FLAG_NORMAL});
                // Double push
                int double_to = (side == WHITE) ? from + 16 : from - 16;
                if (((side == WHITE && from / 8 == 1) || (side == BLACK && from / 8 == 6)) &&
                    !Bitboard::get_bit(occupancies[BOTH], double_to)) {
                    list.add({from, double_to, MOVE_FLAG_DOUBLE_PAWN_PUSH});
                }
            }
        }

        // Captures
        U64 attacks = Attacks::pawn_attacks[side & 1][from] & occupancies[side == WHITE ? BLACK : WHITE];
        while (attacks) {
            to = Bitboard::pop_lsb(attacks);
            if ((side == WHITE && to / 8 == 7) || (side == BLACK && to / 8 == 0)) {
                list.add({from, to, MOVE_FLAG_PROMOTION_QUEEN});
                list.add({from, to, MOVE_FLAG_PROMOTION_ROOK});
                list.add({from, to, MOVE_FLAG_PROMOTION_BISHOP});
                list.add({from, to, MOVE_FLAG_PROMOTION_KNIGHT});
            } else {
                list.add({from, to, MOVE_FLAG_NORMAL});
            }
        }

        // EP
        if (en_passant_sq != -1) {
            U64 ep_attacks = Attacks::pawn_attacks[side & 1][from] & (1ULL << en_passant_sq);
            if (ep_attacks) {
                list.add({from, en_passant_sq, MOVE_FLAG_EN_PASSANT});
            }
        }
    }
}

void Board::generate_leaper_moves(MoveList &list) const {
    // Knights
    U64 knights = bitboards[side == WHITE ? W_KNIGHT : B_KNIGHT];
    while (knights) {
        int from = Bitboard::pop_lsb(knights);
        U64 moves = Attacks::knight_attacks[from] & ~occupancies[side];
        while (moves) {
            list.add({from, Bitboard::pop_lsb(moves), MOVE_FLAG_NORMAL});
        }
    }

    // King
    U64 king = bitboards[side == WHITE ? W_KING : B_KING];
    if (king) {
        int from = Bitboard::pop_lsb(king);
        U64 moves = Attacks::king_attacks[from] & ~occupancies[side];
        while (moves) {
            list.add({from, Bitboard::pop_lsb(moves), MOVE_FLAG_NORMAL});
        }
    }
}

void Board::generate_sliding_moves(MoveList &list) const {
    U64 bishops = bitboards[side == WHITE ? W_BISHOP : B_BISHOP] | bitboards[side == WHITE ? W_QUEEN : B_QUEEN];
    while (bishops) {
        int from = Bitboard::pop_lsb(bishops);
        U64 moves = Magic::get_bishop_attacks(from, occupancies[BOTH]) & ~occupancies[side];
        while (moves) {
            list.add({from, Bitboard::pop_lsb(moves), MOVE_FLAG_NORMAL});
        }
    }

    U64 rooks = bitboards[side == WHITE ? W_ROOK : B_ROOK] | bitboards[side == WHITE ? W_QUEEN : B_QUEEN];
    while (rooks) {
        int from = Bitboard::pop_lsb(rooks);
        U64 moves = Magic::get_rook_attacks(from, occupancies[BOTH]) & ~occupancies[side];
        while (moves) {
            list.add({from, Bitboard::pop_lsb(moves), MOVE_FLAG_NORMAL});
        }
    }
}

void Board::generate_castling_moves(MoveList &list) const {
    if (side == WHITE) {
        if (castle_rights & 1) { // K
            if (!Bitboard::get_bit(occupancies[BOTH], F1) && !Bitboard::get_bit(occupancies[BOTH], G1)) {
                if (!is_square_attacked(E1, BLACK) && !is_square_attacked(F1, BLACK) && !is_square_attacked(G1, BLACK)) {
                    list.add({E1, G1, MOVE_FLAG_CASTLING});
                }
            }
        }
        if (castle_rights & 2) { // Q
            if (!Bitboard::get_bit(occupancies[BOTH], D1) && !Bitboard::get_bit(occupancies[BOTH], C1) && !Bitboard::get_bit(occupancies[BOTH], B1)) {
                if (!is_square_attacked(E1, BLACK) && !is_square_attacked(D1, BLACK) && !is_square_attacked(C1, BLACK)) {
                    list.add({E1, C1, MOVE_FLAG_CASTLING});
                }
            }
        }
    } else {
        if (castle_rights & 4) { // k
            if (!Bitboard::get_bit(occupancies[BOTH], F8) && !Bitboard::get_bit(occupancies[BOTH], G8)) {
                if (!is_square_attacked(E8, WHITE) && !is_square_attacked(F8, WHITE) && !is_square_attacked(G8, WHITE)) {
                    list.add({E8, G8, MOVE_FLAG_CASTLING});
                }
            }
        }
        if (castle_rights & 8) { // q
            if (!Bitboard::get_bit(occupancies[BOTH], D8) && !Bitboard::get_bit(occupancies[BOTH], C8) && !Bitboard::get_bit(occupancies[BOTH], B8)) {
                if (!is_square_attacked(E8, WHITE) && !is_square_attacked(D8, WHITE) && !is_square_attacked(C8, WHITE)) {
                    list.add({E8, C8, MOVE_FLAG_CASTLING});
                }
            }
        }
    }
}
