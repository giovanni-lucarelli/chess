#include "game.hpp"
#include <iostream>

Game::Game() {
    board.reset();
    turn = 0;
}

void Game::play() {
    
    while (true) {

        turn += 1;
        board.print();

        // Check if the king is in check
        board.check_control();
        if(board.get_check(WHITE)) {
            std::cout << "\033[1;31mWhite in check!\033[0m" << std::endl;
        } else if(board.get_check(BLACK)) {
            std::cout << "\033[1;31mBlack in check!\033[0m" << std::endl;
        }

        // printing en passant square
        if(board.get_en_passant_square() != NO_SQUARE) {
            std::cout << "\nEn passant square: " << square_to_string(board.get_en_passant_square()) << std::endl;
        }

        // Checkmate
        if(board.legal_moves(board.get_side_to_move()).size() == 0 && board.get_check(board.get_side_to_move())) {
            Color color = board.get_side_to_move() == WHITE ? BLACK : WHITE;
            std::cout << "\033[1;31mCheckmate!\033[0m" << color << " wins!" << std::endl;
            break;
        }

        // printing color to move in blue and relative turn
        if(board.get_side_to_move() == WHITE) {
            std::cout << "\033[1;34mWhite to move (\033[0m" << (turn / 2) + 1 << "\033[1;34m)\033[0m" << std::endl;
        } else {
            std::cout << "\033[1;34mBlack to move (\033[0m" << (turn / 2) << "\033[1;34m)\033[0m" << std::endl;
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
        std::vector<std::pair<Square, Square>> moves = board.legal_moves(from);
        std::cout << "Legal moves: ";
        for (const auto& move : moves) {
            std::cout << square_to_string(move.second) << " ";
            int n_moves = moves.size();
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
    }

}

std::pair<Square, Square> Game::parse_input(const std::string& from, const std::string& to) const {
    Square from_sq = static_cast<Square>(8 * (from[1] - '1') + (from[0] - 'a'));
    Square to_sq = static_cast<Square>(8 * (to[1] - '1') + (to[0] - 'a'));

    return {from_sq, to_sq};
}

