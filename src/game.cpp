#include "game.hpp"
#include "search.hpp"
#include <iostream>
#include <chrono>

Game::Game() {
    board.reset();
    turn = 0;
}

// void Game::play() {
    
//     while (true) {

//         turn += 1;
//         board.print();

//         // Check if the king is in check
//         board.check_control();
//         if(board.get_check(WHITE)) {
//             std::cout << "\033[1;31mWhite in check!\033[0m" << std::endl;
//         } else if(board.get_check(BLACK)) {
//             std::cout << "\033[1;31mBlack in check!\033[0m" << std::endl;
//         }


//         // printing en passant square
//         if(board.get_en_passant_square() != NO_SQUARE) {
//             std::cout << "\nEn passant square: " << square_to_string(board.get_en_passant_square()) << std::endl;
//         }

//         // Checkmate
//         if(board.legal_moves(board.get_side_to_move()).size() == 0 && board.get_check(board.get_side_to_move())) {
//             Color color = board.get_side_to_move() == WHITE ? BLACK : WHITE;
//             std::cout << "\033[1;31mCheckmate!\033[0m" << color << " wins!" << std::endl;
//             break;
//         }

//         // printing color to move in blue and relative turn
//         if(board.get_side_to_move() == WHITE) {
//             std::cout << "\033[1;34mWhite to move (\033[0m" << (turn / 2) + 1 << "\033[1;34m)\033[0m" << std::endl;
//         } else {
//             std::cout << "\033[1;34mBlack to move (\033[0m" << (turn / 2) << "\033[1;34m)\033[0m" << std::endl;
//         }
        
//         std::string inputPiece;
//         std::string inputMove;
//         std::cout << "Enter piece to move (only its square): ";
//         std::cin >> inputPiece;
//         if(inputPiece == "exit") {
//             break;
//         }

//         Square from = static_cast<Square>(8 * (inputPiece[1] - '1') + (inputPiece[0] - 'a'));

//         // printing legal moves
//         std::vector<std::pair<Square, Square>> moves = board.legal_moves(from);
//         std::cout << "Legal moves: ";
//         for (const auto& move : moves) {
//             std::cout << square_to_string(move.second) << " ";
//         }
//         std::cout << std::endl;


        
//         std::cout << "Enter move (only its square): ";
//         std::cin >> inputMove;
//         if(inputMove == "exit") {
//             break;
//         }

//         std::pair<Square, Square> move = parse_input(inputPiece, inputMove);



//         // Check if the move is legal
//         if (!board.is_move_legal(from, move.second)) {
//             std::cout << "\033[1;31mIllegal move\033[0m\n";
//             continue;
//         }

//         // Make the move
//         board.move_piece(from, move.second);

        

//         // Check if the game is over
//         // if (board.is_checkmate()) {
//         //     display_board();
//         //     std::cout << "Checkmate\n";
//         //     break;
//         // }
//     }

// }

std::pair<Square, Square> Game::parse_input(const std::string& from, const std::string& to) const {
    Square from_sq = static_cast<Square>(8 * (from[1] - '1') + (from[0] - 'a'));
    Square to_sq = static_cast<Square>(8 * (to[1] - '1') + (to[0] - 'a'));

    return {from_sq, to_sq};
}

Move Game::parse_move(const std::string& from, const std::string& to) const {
    Square from_sq = static_cast<Square>(8 * (from[1] - '1') + (from[0] - 'a'));
    Square to_sq = static_cast<Square>(8 * (to[1] - '1') + (to[0] - 'a'));

    // Get the moving piece
    Piece piece = board.get_piece_on_square(from_sq).second;
    Color color = board.get_piece_on_square(from_sq).first;

    Move move(from_sq, to_sq, piece);

    // Check if the move captures a piece
    if (board.is_occupied(to_sq)) {
        move.captured_piece = board.get_piece_on_square(to_sq).second;
    }

    // Handle pawn moves
    if (piece == PAWN) {
        int from_rank = from[1] - '1';
        int to_rank = to[1] - '1';
        
        // Check for promotion (pawn reaches last rank)
        if ((color == WHITE && to_rank == 7) || (color == BLACK && to_rank == 0)) {
            move.type = PROMOTION;
            move.promoted_to = QUEEN;  // Default to queen; should ask the player in an interactive game
        }
        
        // Check for en passant capture
        if (to_sq == board.en_passant_square) {
            move.type = EN_PASSANT;
            move.captured_piece = PAWN;
        }

        // Check for double pawn push (from rank 2 to 4 or from 7 to 5)
        if (std::abs(to_rank - from_rank) == 2) {
            move.type = DOUBLE_PAWN_PUSH;
        }
    }

    // Handle castling
    if (piece == KING && std::abs(from_sq - to_sq) == 2) {
        move.type = CASTLING;
    }

    move.color = color;

    return move;
}


void Game::play_vs_pc() {
    while(true) {
        turn += 1;
        board.print();
        board.check_control();
        if(board.get_check(WHITE)) {
            std::cout << "\033[1;31mWhite in check!\033[0m" << std::endl;
        } else if(board.get_check(BLACK)) {
            std::cout << "\033[1;31mBlack in check!\033[0m" << std::endl;
        }

        if(board.legal_moves(board.get_side_to_move()).size() == 0 && board.get_check(board.get_side_to_move())) {
            Color color = board.get_side_to_move() == WHITE ? BLACK : WHITE;
            std::cout << "\033[1;31mCheckmate!\033[0m" << color << " wins!" << std::endl;
            break;
        }

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
        std::vector<Move> moves = board.legal_moves(from);
        std::cout << "Legal moves: ";

        for (const auto& move : moves) {
            std::cout << square_to_string(move.to) << " ";
        }

        std::cout << std::endl;
        
        std::cout << "Enter move (only its square): ";
        std::cin >> inputMove;
        if(inputMove == "exit") {
            break;
        }

        // ? std::pair<Square, Square> move = parse_input(inputPiece, inputMove);

        Move move = parse_move(inputPiece, inputMove);

        // Check if the move is legal
        if (!board.is_move_legal(move)) {
            std::cout << "\033[1;31mIllegal move\033[0m\n";
            continue;
        }

        // Make the move
        board.do_move(move);

        // Evaluate the position
        int score = evaluate(board);
        std::cout << "Position score: " << score << std::endl;

        // Initialize move generator and search depth
        MoveGenerator movegen{};
        int search_depth = 4;
        std::cout << "Looking for best move...\n";

        // Measure time taken to find best move & find best move
        auto start = std::chrono::high_resolution_clock::now();
        Move best_move = find_best_move(board, search_depth);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken to find best move: " << elapsed.count() << " seconds" << std::endl;
        
        // Print best move
        std::cout << "Best move: " << square_to_string(best_move.from) 
                  << " to " << square_to_string(best_move.to) << std::endl;

    }
}

