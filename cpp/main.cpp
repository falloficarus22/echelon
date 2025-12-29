#include <iostream>
#include <chrono>
#include "board.hpp"
#include "attacks.hpp"
#include "magic.hpp"

U64 perft(Board &board, int depth) {
    if (depth == 0) return 1ULL;

    MoveList list;
    board.generate_moves(list);
    
    U64 nodes = 0;
    for (int i = 0; i < list.count; i++) {
        History hist = board.make_move(list.moves[i]);
        if (!board.is_square_attacked(Bitboard::get_lsb_index(board.bitboards[(1 - board.side) == WHITE ? W_KING : B_KING]), board.side)) {
            nodes += perft(board, depth - 1);
        }
        board.unmake_move(list.moves[i], hist);
    }
    return nodes;
}

int main() {
    // Initialize tables
    Attacks::init_all();
    Magic::init_all();

    Board board;
    board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cout << "Starting Perft Test..." << std::endl;
    for (int depth = 1; depth <= 5; depth++) {
        auto start = std::chrono::high_resolution_clock::now();
        U64 nodes = perft(board, depth);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "Depth " << depth << ": " << nodes << " nodes, Time: " << elapsed.count() << "s, NPS: " << (uint64_t)(nodes / elapsed.count()) << std::endl;
    }

    return 0;
}
