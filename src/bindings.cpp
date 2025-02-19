#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "chessboard.hpp"
#include "game.hpp"
#include "types.hpp"
#include "bitboard.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chessengine_py, m) {
    m.doc() = "Chess engine module bound via pybind11";

    // Bind enums from types.hpp
    py::enum_<Color>(m, "Color")
        .value("WHITE", WHITE)
        .value("BLACK", BLACK)
        .value("NO_COLOR", NO_COLOR)
        .export_values();

    py::enum_<Piece>(m, "Piece")
        .value("PAWN", PAWN)
        .value("KNIGHT", KNIGHT)
        .value("BISHOP", BISHOP)
        .value("ROOK", ROOK)
        .value("QUEEN", QUEEN)
        .value("KING", KING)
        .value("NO_PIECE", NO_PIECE)
        .export_values();

    py::enum_<Square>(m, "Square")
        .value("A1", A1)
        .value("B1", B1)
        .value("C1", C1)
        .value("D1", D1)
        .value("E1", E1)
        .value("F1", F1)
        .value("G1", G1)
        .value("H1", H1)
        .value("A2", A2)
        .value("B2", B2)
        .value("C2", C2)
        .value("D2", D2)
        .value("E2", E2)
        .value("F2", F2)
        .value("G2", G2)
        .value("H2", H2)
        // ... add all squares up to H8
        .value("H8", H8)
        .value("NO_SQUARE", NO_SQUARE)
        .export_values();

    // Bind the ChessBoard class.
    py::class_<ChessBoard>(m, "ChessBoard")
        .def(py::init<>())
        .def("reset", &ChessBoard::reset)
        .def("print", &ChessBoard::print)
        .def("get_pieces", &ChessBoard::get_pieces)
        .def("get_piece_on_square", &ChessBoard::get_piece_on_square)
        .def("remove_piece", &ChessBoard::remove_piece)
        .def("add_piece", &ChessBoard::add_piece)
        .def("move_piece", &ChessBoard::move_piece, py::arg("from"), py::arg("to"), py::arg("interactive") = true)
        .def("is_path_clear", &ChessBoard::is_path_clear)
        .def("is_occupied", &ChessBoard::is_occupied)
        .def("pseudo_legal_targets", &ChessBoard::pseudo_legal_targets)
        .def("is_move_legal", &ChessBoard::is_move_legal)
        .def("legal_moves", (std::vector<std::pair<Square, Square>> (ChessBoard::*)(Square) const) &ChessBoard::legal_moves)
        .def("legal_moves_color", (std::vector<std::pair<Square, Square>> (ChessBoard::*)(Color) const) &ChessBoard::legal_moves)
        .def("check_control", &ChessBoard::check_control)
        .def("choose_promotion_piece", &ChessBoard::choose_promotion_piece)
        // Optionally, add getters for en passant or side-to-move if needed.
        ;

    // Bind the Game class.
    py::class_<Game>(m, "Game")
        .def(py::init<>())
        .def("play", &Game::play)
        .def("parse_input", &Game::parse_input)
        ;
}



