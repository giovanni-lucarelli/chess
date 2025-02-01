#pragma once
#include "types.hpp"
#include <string>

struct Move {
    Square from;
    Square to;
    Piece promotion;
    bool is_en_passant;
    bool is_capture;
    bool is_double_pawn_push;
    bool is_pawn_move;
    bool is_check;
    bool is_checkmate;
    bool is_stalemate;
    bool is_draw;
    bool is_promotion;
    bool is_kingside_castling;
    bool is_queenside_castling;

    // Constructor
    Move(Square from, Square to, Piece promotion = PAWN, bool is_en_passant = false, bool is_capture = false, bool is_double_pawn_push = false, bool is_pawn_move = false, bool is_check = false, bool is_checkmate = false, bool is_stalemate = false, bool is_draw = false, bool is_promotion = false, bool is_kingside_castling = false, bool is_queenside_castling = false)
        : from(from), to(to), promotion(promotion), is_en_passant(is_en_passant), is_capture(is_capture), is_double_pawn_push(is_double_pawn_push), is_pawn_move(is_pawn_move), is_check(is_check), is_checkmate(is_checkmate), is_stalemate(is_stalemate), is_draw(is_draw), is_promotion(is_promotion), is_kingside_castling(is_kingside_castling), is_queenside_castling(is_queenside_castling) {}

    // to_string method
    std::string to_string() const {
        return "Move from " + square_to_string(from) + " to " + square_to_string(to);
    }
};
