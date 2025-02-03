#include "game.hpp"
#include <iostream>

void Game::start1v1() {
    board.reset();
    
    while (true) {
        display_board();
        
        // this is important for the Computer to know the current board state
        // and the avaiable moves so it can make the best one.
        // std::vector<Move> legal_moves = MoveGenerator::generate_moves(board);
        // if (legal_moves.empty()) {
        //     std::cout << "Game over\n";
        //     break;
        // }

        // Get user input in algebraic notation (e.g. e2e4)
        std::string inputPiece;
        std::cout << "Enter piece to move (only its square): ";
        std::cin >> inputPiece;
        if(inputPiece == "exit") {
            break;
        }
        std::string inputMove;
        std::cout << "Enter move (only its square): ";
        std::cin >> inputMove;
        if(inputMove == "exit") {
            break;
        }

        // Concatenating
        std::string input = inputPiece + inputMove;
        Move move = parse_input(input);

        if (board.is_move_legal(move.from, move.to)) {
            board_history.push(board); // Save current board state before making a move
            board.move_piece(move.from, move.to);
        } else {
            std::cout << "Illegal move. Try again.\n";
        }
    }
}

void Game::display_board() const {
    board.print();
}

Move Game::parse_input(const std::string& input) const {
    if (input.size() != 4) {
        return Move(Square::NO_SQUARE, Square::NO_SQUARE); // Invalid move
    }

    Square from = static_cast<Square>(8 * (input[1] - '1') + (input[0] - 'a'));
    Square to = static_cast<Square>(8 * (input[3] - '1') + (input[2] - 'a'));

    return Move(from, to);
}
