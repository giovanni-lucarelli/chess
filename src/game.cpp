#include "game.hpp"
#include <iostream>

#include <stack>

void Game::start() {
    std::stack<ChessBoard> board_history; // Stack to store board states
    board.reset();
    
    while (true) {
        display_board();
        std::vector<Move> legal_moves = MoveGenerator::generate_moves(board);
        
        std::cout << "Legal moves:\n";
        for (const Move& move : legal_moves) {
            std::cout << move.to_string() << "\n";
        }

        std::string input;
        std::cin >> input;
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
    // Implement input parsing logic here
    // For now, return a dummy move with all required arguments
    return Move(A2, A3, PAWN, false, false, false, true, false, false, false, false, false, false, false);
}
