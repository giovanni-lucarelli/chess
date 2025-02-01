#pragma once
#include "types.hpp"
#include <string>

struct Move {
    Square from;
    Square to;
    // Piece promotion;
    // bool is_en_passant;
    // bool is_capture;
    // bool is_double_pawn_push;
    // bool is_pawn_move;
    // bool is_check;
    // bool is_checkmate;
    // bool is_stalemate;
    // bool is_draw;
    // bool is_promotion;
    // bool is_kingside_castling;
    // bool is_queenside_castling;

    // Constructor
    Move(Square from, Square to) : from(from), to(to) {
    }
    
    // to_string method
    std::string to_string() const {
        return "Move from " + square_to_string(from) + " to " + square_to_string(to);
    }
};
