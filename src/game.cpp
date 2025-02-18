#include "game.hpp"
#include <iostream>

Game::Game() {
    board.reset();
}

void Game::play() {
    
    while (true) {
        board.print();

        // printing color to move in blue
        if(board.get_side_to_move() == WHITE) {
            std::cout << "\033[1;34mWhite to move\033[0m" << std::endl;
        } else {
            std::cout << "\033[1;34mBlack to move\033[0m" << std::endl;
        }

        // Check if the king is in check
        board.check_control();
        if(board.get_check(board.get_side_to_move())) {
            std::cout << "\033[1;31mCheck!\033[0m" << std::endl;
        }
        if(board.get_check(board.get_side_to_move() == WHITE ? BLACK : WHITE)) {
            std::cout << "\033[1;31mCheck!\033[0m" << std::endl;
        }
        
        std::string inputPiece;
        std::string inputMove;
        std::cout << "Enter piece to move (only its square): ";
        std::cin >> inputPiece;
        if(inputPiece == "exit") {
            break;
        }

        Square from = static_cast<Square>(8 * (inputPiece[1] - '1') + (inputPiece[0] - 'a'));

        // printing legal moves
        std::vector<Square> legal_moves = board.pseudo_legal_targets(from);
        std::cout << "Legal moves: ";
        for (Square sq : legal_moves) {
            std::cout << static_cast<char>('a' + sq % 8) << static_cast<char>('1' + sq / 8) << " ";
        }
        std::cout << std::endl;


        
        std::cout << "Enter move (only its square): ";
        std::cin >> inputMove;
        if(inputMove == "exit") {
            break;
        }

        std::pair<Square, Square> move = parse_input(inputPiece, inputMove);

        // Obtaining piece and color
        std::pair<Color, Piece> piece_info = board.get_piece_on_square(from);


        // Check if the move is legal
        if (!board.is_move_legal(from, move.second)) {
            std::cout << "Illegal move\n";
            continue;
        }

        // Make the move
        board.move_piece(from, move.second);

        // Check if the game is over
        // if (board.is_checkmate()) {
        //     display_board();
        //     std::cout << "Checkmate\n";
        //     break;
        // }

        turn += 1;
    }
}

std::pair<Square, Square> Game::parse_input(const std::string& from, const std::string& to) const {
    Square from_sq = static_cast<Square>(8 * (from[1] - '1') + (from[0] - 'a'));
    Square to_sq = static_cast<Square>(8 * (to[1] - '1') + (to[0] - 'a'));

    return {from_sq, to_sq};
}

