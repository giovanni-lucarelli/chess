#include "game.hpp"
#include <iostream>

void Game::start1v1() {
    board.reset();
    
    while (true) {
        display_board();

        // Check if the king is in check
        board.check_control();
        if(board.get_check(board.get_side_to_move())) {
            std::cout << "\033[1;31mCheck!\033[0m" << std::endl;
        }
        if(board.get_check(board.get_side_to_move() == WHITE ? BLACK : WHITE)) {
            std::cout << "\033[1;31mCheck!\033[0m" << std::endl;
        }
        
        // this is important for the Computer to know the current board state
        // and the avaiable moves so it can make the best one.
        // std::vector<Move> legal_moves = MoveGenerator::generate_moves(board);
        // if (legal_moves.empty()) {
        //     std::cout << "Game over\n";
        //     break;
        // }
        std::string inputPiece;
        std::string inputMove;
        std::cout << "Enter piece to move (only its square): ";
        std::cin >> inputPiece;
        if(inputPiece == "exit") {
            break;
        }
        
        
        Square from = parse_single_input(inputPiece);
        std::pair<Color, Piece> piece_from = board.get_piece_on_square(from);

        // printing legal moves
        std::vector<Move> legal_moves = board.legal_moves(from);
        std::cout << "\033[1;32mlegal moves:\033[0m" << std::endl;
        for (auto move : legal_moves) {
            std::cout << square_to_string(move.to) << std::endl;
        }


        
        std::cout << "Enter move (only its square): ";
        std::cin >> inputMove;
        if(inputMove == "exit") {
            break;
        }

        // Concatenating
        std::string input = inputPiece + inputMove;
        Move move = parse_input(input);

        // Obtaining piece and color
        std::pair<Color, Piece> piece = board.get_piece_on_square(move.from);

        if (board.is_move_legal(move.from, move.to)) {
            board_history.push(board); // Save current board state before making a move
            board.move_piece(move.from, move.to);
        } else {
            std::cout << "Illegal move. Try again.\n";
        }

        // Check if the game is over
        // if (board.is_checkmate()) {
        //     display_board();
        //     std::cout << "Checkmate\n";
        //     break;
        // }

        turn += 1;
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

// Parse single input
Square Game::parse_single_input(const std::string& input) const {
    if (input.size() != 2) {
        return Square::NO_SQUARE; // Invalid move
    }

    Square sq = static_cast<Square>(8 * (input[1] - '1') + (input[0] - 'a'));

    return sq;
}
