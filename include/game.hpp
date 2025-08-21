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
    void set_board(ChessBoard board){ this->board = board; }
    void update_check();
    
    /* ------------------------------- Parse Input ------------------------------ */
    
    // std::pair<Square, Square> parse_input(const std::string& from, const std::string& to) const;
    Move parse_move(const std::string& from, const std::string& to) const;
    Move parse_action_to_move(int action) const;

    void reset_from_fen(const std::string& fen);
    std::string to_fen() const;
    
    /* --------------------------------- Utility -------------------------------- */
    
    bool is_move_legal(Move move) const;
    // ? is_checkmate = is_game_over ?
    // bool is_game_over();
    // ? what is this doing ?

    // /// Returns true if *side_to_move()* is currently in check
    // bool in_check() const {
    //     // make sure your check_control() has been called most recently
    //     // or call it here if it doesn’t mutate anything else 
    //     return (get_side_to_move() == WHITE ? white_check : black_check);
    // }

    // Returns (white_in_check, black_in_check) without mutating state
    std::pair<bool,bool> compute_check_flags() const;

    bool in_check() const {
        auto [w_check, b_check] = compute_check_flags();
        return get_side_to_move() == WHITE ? w_check : b_check;
    }

    /// True if side to move is in check and has no legal moves
    bool is_checkmate() const {
        return in_check() && legal_moves(get_side_to_move()).empty();
    }

    bool is_insufficient_material() const;

    /// True if side to move is *not* in check but has no legal moves
    bool is_stalemate() const {
        return !in_check() && legal_moves(get_side_to_move()).empty();
    }

    bool is_draw() const {
        return is_stalemate() || is_insufficient_material()
            /* || is_threefold() || is_fifty_move_rule() */;
    }

    /// Game is over if it’s a checkmate *or* a stalemate
    bool is_game_over() const {
        // terminal by rule
        if (is_checkmate()) return true;
        if (is_stalemate()) return true;
        if (is_insufficient_material()) return true;      // if you implement this
        // if (is_fifty_move_rule()) return true;            // halfmove clock >= 100
        // if (is_threefold_repetition()) return true;       // if tracked

        return false;
    }

    /// +1 for White win, -1 for Black win, 0 otherwise
    double result() const {
        if (!is_game_over()) 
            return 0.0;
        if (is_checkmate()) 
            // if it’s checkmate, the *side to move* just lost
            return (get_side_to_move() == WHITE ? -1.0 : +1.0);
        // stalemate ⇒ draw
        return 0.0;
    }
    
    /* --------------------------------- Actions -------------------------------- */
    
    void do_move(Move& move);
    void undo_move(Move& move);
    virtual Piece choose_promotion_piece() const;

    std::vector<Move> pseudo_legal_moves(Square from) const;
    std::vector<Move> legal_moves(Square from) const;
    std::vector<Move> legal_moves(Color color) const;
    
    void play();    
};