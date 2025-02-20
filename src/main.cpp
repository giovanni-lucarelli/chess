#include "game.hpp"

void play() {
    int turn = 0;
    Game game;
    
    while (true) {

        turn += 1;
        game.board.print();
        // Check if the king is in check
        game.board.check_control();
        if(game.board.get_check(WHITE)) {
            std::cout << "\033[1;31mWhite in check!\033[0m" << std::endl;
        } else if(game.board.get_check(BLACK)) {
            std::cout << "\033[1;31mBlack in check!\033[0m" << std::endl;
        }


        // printing en passant square
        if(game.board.get_en_passant_square() != NO_SQUARE) {
            std::cout << "\nEn passant square: " << square_to_string(game.board.get_en_passant_square()) << std::endl;
        }

        // Checkmate
        if(game.board.legal_moves(game.board.get_side_to_move()).size() == 0 && game.board.get_check(game.board.get_side_to_move())) {
            Color color = game.board.get_side_to_move() == WHITE ? BLACK : WHITE;
            std::cout << "\033[1;31mCheckmate!\033[0m" << color << " wins!" << std::endl;
            break;
        }

        // printing color to move in blue and relative turn
        if(game.board.get_side_to_move() == WHITE) {
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
        std::vector<std::pair<Square, Square>> moves = game.board.legal_moves(from);
        std::cout << "Legal moves: ";
        for (const auto& move : moves) {
            std::cout << square_to_string(move.second) << " ";
        }
        std::cout << std::endl;


        
        std::cout << "Enter move (only its square): ";
        std::cin >> inputMove;
        if(inputMove == "exit") {
            break;
        }

        std::pair<Square, Square> move = game.parse_input(inputPiece, inputMove);



        // Check if the move is legal
        if (!game.board.is_move_legal(from, move.second)) {
            std::cout << "Illegal move\n";
            continue;
        }

        // Make the move
        game.board.move_piece(from, move.second);

    
    }

}

int main() {
    Game game;
    play();
    return 0;
}
