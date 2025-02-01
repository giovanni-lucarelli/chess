// chessboard.hpp
#pragma once
#include "types.hpp"
#include "move.hpp"
#include <array>
#include <string>
#include <vector>

class ChessBoard {
private:

    // Bitboards for [Color][Piece]
    std::array<std::array<U64, 6>, 2> pieces;

    // Game state
    Color side_to_move;
    Square en_passant_square; // Use Square::H8 or similar as "no square"
    bool castling_rights[2][2]; // [Color][Queenside, Kingside]
    // int halfmove_clock;
    // int fullmove_number;

public:

    ChessBoard();

    // Initialize classic starting position
    void reset();

    // Get all pieces of a type/color
    U64 get_pieces(Color color, Piece piece) const;

    // Check if a square is occupied by any piece
    bool is_occupied(Square sq) const;

    // return the piece and the color on a square
    std::pair<Color, Piece> get_piece_on_square(Square sq) const;

    // Print board (for debugging)
    void print() const;

    // Getters
    Color get_side_to_move() const { return side_to_move; }
    Square get_en_passant_square() const { return en_passant_square; }
    bool get_castling_rights(Color color, bool kingside) const { return castling_rights[color][kingside]; }

    // Setters
    void set_side_to_move(Color color) { side_to_move = color; }
    void set_en_passant_square(Square sq) { en_passant_square = sq; }
    void set_castling_rights(Color color, bool kingside, bool value) { castling_rights[color][kingside] = value; }

    // remove a piece
    void remove_piece(Square sq);

    // add a piece
    void add_piece(Color color, Piece piece, Square sq);

    // check legality of a move
    bool is_move_legal(Square from, Square to) const;

    // make a move
    void move_piece(Square from, Square to);

    std::vector<Move> generate_legal_moves() const;

};

// Utility function to conver a number to each color or piece
inline std::string color_to_string(Color c) {
  return c == WHITE ? "WHITE" : "BLACK";
}

inline std::string piece_to_string(Piece p) {
  switch (p) {
    case PAWN: return "PAWN";
    case KNIGHT: return "KNIGHT";
    case BISHOP: return "BISHOP";
    case ROOK: return "ROOK";
    case QUEEN: return "QUEEN";
    case KING: return "KING";
    default: return "INVALID";
  }
}
