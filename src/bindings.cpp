#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game.hpp"
#include "move.hpp"
#include "chessboard.hpp"
#include "types.hpp"  // Assuming Color is defined here

namespace py = pybind11;

PYBIND11_MODULE(chessengine_py, m) {
    m.doc() = "Python bindings for the Chess Engine project"; // Module documentation

    // Bind the Color enum.
    py::enum_<Color>(m, "Color")
        .value("NO_COLOR", NO_COLOR)
        .value("WHITE", WHITE)
        .value("BLACK", BLACK)
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

    // square to string
    m.def("square_to_string", &square_to_string);

    // piece to string
    m.def("piece_to_string", &piece_to_string);

    py::class_<Game>(m, "Game")
        .def(py::init<>()) // Constructor
        .def("play", &Game::play, "Play a game of chess")
        .def("play_vs_pc", &Game::play_vs_pc, "Play against the computer")
        .def("is_game_over", &Game::is_game_over, "Check if the game is over")
        .def("get_board", &Game::get_board, "Get the current board state")
        .def("get_side_to_move", &Game::get_side_to_move, "Get the color of the side to move")
        .def("get_en_passant_square", &Game::get_en_passant_square, "Get the en passant square")
        .def("set_en_passant_square", &Game::set_en_passant_square, "Set the en passant square")
        .def("choose_promotion_piece", &Game::choose_promotion_piece, "Choose a piece for pawn promotion")
        .def("is_move_legal", &Game::is_move_legal, "Check if a move is legal")
        .def("do_move", &Game::do_move, "Make a move on the board")
        .def("undo_move", &Game::undo_move, "Undo the last move")
        .def("legal_moves", py::overload_cast<Square>(&Game::legal_moves, py::const_), "Get legal moves from a square")
        .def("legal_moves_color", py::overload_cast<Color>(&Game::legal_moves, py::const_), "Get legal moves for a color")  
        .def("get_check", &Game::get_check, "Check if a color is in check")
        .def("check_control", &Game::check_control, "Check control of the board");

    py::class_<Move>(m, "Move")
        .def(py::init<>()) // Default constructor
        .def(py::init([](Color color, Piece piece, Square from, Square to) {
             // Create Move using the constructor that requires 
             // (Color, Piece, Square, Square, MoveType, Piece, Piece)
             // Defaulting MoveType to NORMAL and captured/promoted pieces to NO_PIECE.
             return Move(color, piece, from, to, MoveType::NORMAL, NO_PIECE, NO_PIECE);
         }), "Constructor with parameters: (Color, Piece, Square, Square)")
        .def("print", &Move::print, "Print the move details")
        .def_readwrite("color", &Move::color)
        .def_readwrite("piece", &Move::piece)
        .def_readwrite("from", &Move::from)
        .def_readwrite("to", &Move::to)
        .def_readwrite("type", &Move::type)
        .def_readwrite("captured_piece", &Move::captured_piece)
        .def_readwrite("promoted_to", &Move::promoted_to);
    // Add other methods and properties as needed

    py::class_<ChessBoard>(m, "ChessBoard")
        .def(py::init<>()) // Constructor
        .def("get_board", &ChessBoard::get_board, "Get the current board state")
        .def("print", &ChessBoard::print, "Print the board")
        .def("reset", &ChessBoard::reset, "Reset the board")
        .def("get_pieces", &ChessBoard::get_pieces, "Get pieces of a specific color and type")
        .def("is_path_clear", &ChessBoard::is_path_clear, "Check if the path between two squares is clear")
        .def("is_occupied", &ChessBoard::is_occupied, "Check if a square is occupied");
    // Add other methods and properties as needed

    py::class_<Bitboard>(m, "Bitboard")
        .def_static("set_bit", &Bitboard::set_bit, "Set a bit at a specific square")
        .def_static("clear_bit", &Bitboard::clear_bit, "Clear a bit at a specific square")
        .def_static("get_bit", &Bitboard::get_bit, "Get the value of a bit at a specific square")
        .def_static("count_bits", &Bitboard::count_bits, "Count the number of bits set in a bitboard")
        .def_static("print", &Bitboard::print, "Print the bitboard with rank/file labels");
    // Add other methods and properties as needed

    // Add any additional bindings for enums or other classes as needed
}