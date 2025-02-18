#pragma once
#include "types.hpp"
#include "move.hpp"
#include "bitboard.hpp"
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
    bool white_check = false;
    bool black_check = false;
    bool checkmate = false;

    bool DEBUG = true;


public:

    ChessBoard();

    // Initialize classic starting position
    void reset();

    bool is_path_clear(Square from, Square to) const;

    // Check if a square is occupied by any piece
    bool is_occupied(Square sq) const;


    // Print board (for debugging)
    void print() const;

    // Getters

    // Get all pieces of a type/color
    U64 get_pieces(Color color, Piece piece) const;

    // return the piece and the color on a square
    std::pair<Color, Piece> get_piece_on_square(Square sq) const;
    Color get_side_to_move() const { return side_to_move; }
    Square get_en_passant_square() const { return en_passant_square; }
    bool get_castling_rights(Color color, bool kingside) const { return castling_rights[color][kingside]; }
    bool get_check(Color color) {
        if (color == WHITE) {
            return white_check;
        } else {
            return black_check;
        }
    }

    // Setters
    void set_side_to_move(Color color) { side_to_move = color; }
    void set_en_passant_square(Square sq) { en_passant_square = sq; }
    void set_castling_rights(Color color, bool kingside, bool value) { castling_rights[color][kingside] = value; }

    // remove a piece
    void remove_piece(Square sq);

    // add a piece
    void add_piece(Color color, Piece piece, Square sq);

    // Move a piece
    void move_piece(Square from, Square to);

    // Pseudo-legal moves
    std::vector<Square> pseudo_legal_targets(Square from) const;

    // Check if a given color's king is in check
    void check_control();

    // Check if a move is legal
    bool is_move_legal(Square from, Square to) const;



    // Legal moves
    std::vector<Move> legal_moves(Square from) const;

};
