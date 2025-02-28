#pragma once
#include "chessboard.hpp"
#include "move.hpp"
#include <string>
#include <stack>

class Move;

class Game {
private:

    ChessBoard board;
    // std::vector<ChessBoard> board_history;
    // std::stack<std::set<Square>> move_history;

    int turn;
    
    // Game state
    Color side_to_move; // turn
    Square en_passant_square; // Use Square::H8 or similar as "no square"
    bool castling_rights[2][2]; // [Color][Queenside, Kingside]
    bool white_check = false;
    bool black_check = false;

    // ? is this triggered by is_game_over ?
    bool checkmate = false;

public:
    
    Game();

    /* --------------------------------- Getters -------------------------------- */
    ChessBoard get_board() const { return board; }
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


    /* ------------------------------ Setters --------------------------------- */

    void set_side_to_move(Color color) { side_to_move = color; }
    void set_en_passant_square(Square sq) { en_passant_square = sq; }
    void set_castling_rights(Color color, bool kingside, bool value) { castling_rights[color][kingside] = value; };
    

    /* ------------------------------- Parse Input ------------------------------ */
    
    // std::pair<Square, Square> parse_input(const std::string& from, const std::string& to) const;
    Move parse_move(const std::string& from, const std::string& to) const;
    
    /* --------------------------------- Utility -------------------------------- */
    
    bool is_move_legal(Move move) const;
    // ? is_checkmate = is_game_over ?
    bool is_game_over();
    // ? what is this doing ?
    void check_control();
    
    /* --------------------------------- Actions -------------------------------- */
    
    void do_move(Move& move);
    void undo_move(Move& move);
    Piece choose_promotion_piece() const;

    std::vector<Move> pseudo_legal_moves(Square from) const;
    std::vector<Move> legal_moves(Square from) const;
    std::vector<Move> legal_moves(Color color) const;
    
    void play();
    void play_vs_pc(const int search_depth);
    
};