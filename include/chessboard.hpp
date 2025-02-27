#pragma once
#include "types.hpp"
#include "bitboard.hpp"
#include <array>
#include <string>
#include <vector>
#include <set>

class ChessBoard {
private:

    // Bitboards for [Color][Piece]
    std::array<std::array<U64, 6>, 2> pieces;

    // Game state
    Color side_to_move;
    Square en_passant_square; // Use Square::H8 or similar as "no square"
    bool castling_rights[2][2]; // [Color][Queenside, Kingside]
    bool white_check = false;
    bool black_check = false;

    // ? is this triggered by is_game_over ?
    bool checkmate = false;

    bool DEBUG = true;


public:
    friend class Game;
    friend struct Move;
    ChessBoard();
    ChessBoard(const ChessBoard& other);
    std::vector<std::vector<std::string>> get_board() const;
    void print() const;
    void reset();

//     /* --------------------------------- */
//     /*              Getters              */
//     /* --------------------------------- */

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


//     /* --------------------------------- */
//     /*              Setters              */
//     /* --------------------------------- */

    void set_side_to_move(Color color) { side_to_move = color; }
    void set_en_passant_square(Square sq) { en_passant_square = sq; }
    void set_castling_rights(Color color, bool kingside, bool value) { castling_rights[color][kingside] = value; };
    Piece choose_promotion_piece() const;

/* -------------------------------------------------------------------------- */
/*                             Checkers and Rules                             */
/* -------------------------------------------------------------------------- */

    bool is_path_clear(Square from, Square to) const;
    bool is_occupied(Square sq) const;
    bool is_move_legal(Move move) const;
    // ? is_checkmate = is_game_over ?
    bool is_game_over();
    // ? what is this doing ?
    void check_control();

    std::vector<Move> pseudo_legal_moves(Square from) const;
    std::vector<Move> legal_moves(Square from) const;
    std::vector<Move> legal_moves(Color color) const;

    // bool is_move_legal(Square from, Square to) const;
    // std::set<Square> pseudo_legal_targets(Square from) const;
    // std::vector<std::pair<Square, Square>> legal_moves(Square from) const;
    // std::vector<std::pair<Square, Square>> legal_moves(Color color) const;
    
    
/* -------------------------------------------------------------------------- */
/*                               Piece Movement                               */
/* -------------------------------------------------------------------------- */

    void remove_piece(Square sq);
    void add_piece(Color color, Piece piece, Square sq);

    void do_move(const Move& move);
    void undo_move(const Move& move, bool interactive);
    
};