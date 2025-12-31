#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "board.hpp"
#include "attacks.hpp"
#include "magic.hpp"
#include "mcts.hpp"


namespace py = pybind11;

PYBIND11_MODULE(echelon_cpp, m) {
    m.doc() = "Echelon Chess Engine C++ Backend";

    // MoveFlag enum
    py::enum_<MoveFlag>(m, "MoveFlag")
        .value("NORMAL", MOVE_FLAG_NORMAL)
        .value("PROMOTION_KNIGHT", MOVE_FLAG_PROMOTION_KNIGHT)
        .value("PROMOTION_BISHOP", MOVE_FLAG_PROMOTION_BISHOP)
        .value("PROMOTION_ROOK", MOVE_FLAG_PROMOTION_ROOK)
        .value("PROMOTION_QUEEN", MOVE_FLAG_PROMOTION_QUEEN)
        .value("DOUBLE_PAWN_PUSH", MOVE_FLAG_DOUBLE_PAWN_PUSH)
        .value("EN_PASSANT", MOVE_FLAG_EN_PASSANT)
        .value("CASTLING", MOVE_FLAG_CASTLING)
        .export_values();

    // History struct
    py::class_<History>(m, "History")
        .def_readwrite("en_passant_sq", &History::en_passant_sq)
        .def_readwrite("castle_rights", &History::castle_rights)
        .def_readwrite("halfmove_clock", &History::halfmove_clock)
        .def_readwrite("captured_piece", &History::captured_piece);

    // Initialize all tables on module import
    m.def("init", []() {
        Attacks::init_all();
        Magic::init_all();
    });

    // Move class - use readable property names
    py::class_<Move>(m, "Move")
        .def(py::init<int, int, MoveFlag>())
        .def_readonly("from_sq", &Move::from)  // Changed from "source"
        .def_readonly("to_sq", &Move::to)       // Changed from "target"
        .def_readonly("flag", &Move::flag);

    // Board class
    py::class_<Board>(m, "BoardState")
        .def(py::init<>())
        .def("parse_fen", &Board::parse_fen)
        .def("make_move", &Board::make_move)
        .def("unmake_move", &Board::unmake_move)
        .def("evaluate", &Board::evaluate)
        .def("is_in_check", [](Board &self, int side) { return self.is_in_check(side); }, py::arg("side") = -1)
        .def("tensorize", [](Board &self) {
            std::vector<float> data = self.tensorize();
            return py::array_t<float>({13, 8, 8}, data.data());
        })
        .def("generate_legal_moves", [](Board &self) {
            MoveList list;
            self.generate_moves(list);
            
            std::vector<Move> legal_moves;
            for (int i = 0; i < list.count; i++) {
                History hist = self.make_move(list.moves[i]);
                // Check if king is attacked after move
                int king_sq = Bitboard::get_lsb_index(self.bitboards[(1 - self.side) == WHITE ? W_KING : B_KING]);
                if (!self.is_square_attacked(king_sq, self.side)) {
                    legal_moves.push_back(list.moves[i]);
                }
                self.unmake_move(list.moves[i], hist);
            }
            return legal_moves;
        });

    // MCTS class
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<int, float, float, float, float>(),
             py::arg("num_simulations") = 800,
             py::arg("c_puct") = 1.5,
             py::arg("temperature") = 1.0,
             py::arg("dirichlet_alpha") = 0.3,
             py::arg("dirichlet_epsilon") = 0.25)
        .def("search", &MCTS::search)
        .def("encode_move", &MCTS::encode_move_for_nn)
        .def("set_temperature", &MCTS::set_temperature);
}
