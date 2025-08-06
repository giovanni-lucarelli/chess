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

    // Determine move type (simplified for 5x5 endgame)
    // No castling, en passant, or promotions in our endgame setup
    if (board.is_occupied(to) && board.get_piece_on_square(to).first != color) {
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

    // No promotions in our endgame setup
    promoted_to = NO_PIECE;

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