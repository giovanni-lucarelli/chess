#include "move.hpp"
#include "game.hpp"  

// Parametrized with automatic MoveType
Move::Move(Color color, Piece piece, Square from, Square to, const Game& game) : 
    color(color), piece(piece), from(from), to(to) {

    auto board = game.get_board();

    // Debugging

    // std::cout << "Move from " << square_to_string(from) << " to " << square_to_string(to) << std::endl;
    // std::cout << "Board occupied at " << square_to_string(to) << ": " << board.is_occupied(to) << std::endl;
    // if (board.is_occupied(to)) {
    //     std::cout << "Piece at destination: " << board.get_piece_on_square(to).second << std::endl;
    //     std::cout << "Color at destination: " << board.get_piece_on_square(to).first << std::endl;
    //     std::cout << "Current move color: " << color << std::endl;
    // }

    // Determine move type
    if (board.get_piece_on_square(from).second == KING) {
        if ((from == E1 && to == G1) || (from == E1 && to == C1) || 
            (from == E8 && to == G8) || (from == E8 && to == C8)) { 
            type = MoveType::CASTLING;
        }
    } else if (board.get_piece_on_square(from).second == PAWN) {
        if (to == game.get_en_passant_square()) {
            type = MoveType::EN_PASSANT;
        } else if (to / 8 == 0 || to / 8 == 7) {
            type = MoveType::PROMOTION;
        } else if ((from / 8 == 1 && to / 8 == 3) || (from / 8 == 6 && to / 8 == 4)) {
            type = MoveType::DOUBLE_PAWN_PUSH;
        } else if (board.is_occupied(to) && board.get_piece_on_square(to).first != color){
            type = MoveType::CAPTURE;
        } else {
            type = MoveType::NORMAL;   
        }
    } else if (board.is_occupied(to) && board.get_piece_on_square(to).first != color) {
        type = MoveType::CAPTURE;
    } else {
        type = MoveType::NORMAL;
    }

    // Determine captured piece (only if it's a capture)
    if (type == MoveType::CAPTURE) {
        captured_piece = board.get_piece_on_square(to).second;
    } else {
        captured_piece = NO_PIECE;
    }

    // Determine promoted piece
    if (type == MoveType::PROMOTION) {
        promoted_to = game.choose_promotion_piece();
    } else {
        promoted_to = NO_PIECE;
    }

};

void Move::print() {
    std::cout << "Move: " << color_to_string(color) << " " << piece_to_string(piece) << " " << square_to_string(from) << " " << square_to_string(to) << std::endl;
    std::cout << "Type: " << movetype_to_string(type) << std::endl;
    if (type == MoveType::CAPTURE) {
        std::cout << "Captured piece: " << piece_to_string(captured_piece) << std::endl;
    }
    if (type == MoveType::PROMOTION) {
        std::cout << "Promoted to: " << piece_to_string(promoted_to) << std::endl;
    }
};