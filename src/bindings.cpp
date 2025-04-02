#include <pybind11/pybind11.h>
#include "game.hpp" // Assuming Game is defined in include/game.hpp

namespace py = pybind11;

PYBIND11_MODULE(chessengine_py, m) {
    m.doc() = "Python bindings for the Chess Engine project"; // Module documentation

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
        .def("undo_move", &Game::undo_move, "Undo the last move");
    // Add other methods and properties as needed

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