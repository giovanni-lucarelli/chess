#include "game.hpp"
#include "types.hpp"
#include "move.hpp"
#include "chessboard.hpp"
#include "search.hpp"
#include <iostream>
#include <chrono>

Game::Game() {
    board.reset();
    turn = 0;
    // Game state
    side_to_move = WHITE;
    castling_rights[WHITE][0] = castling_rights[WHITE][1] = true; // QK
    castling_rights[BLACK][0] = castling_rights[BLACK][1] = true;
    en_passant_square = NO_SQUARE; 
}

Piece Game::choose_promotion_piece() const {
    std::cout << "Promote pawn to (q, r, b, n): ";
    char choice;
    do {
        std::cin >> choice;
    } while (std::tolower(choice) != 'q' && std::tolower(choice) != 'r' &&
             std::tolower(choice) != 'b' && std::tolower(choice) != 'n');
    switch (std::tolower(choice)) {
        case 'q': return QUEEN;
        case 'r': return ROOK;
        case 'b': return BISHOP;
        case 'n': return KNIGHT;
        default: return QUEEN; // default promotion to Queen if input is invalid
    }
}

bool Game::is_game_over() {
    // Check if the side to move is in checkmate
    check_control();
    if (side_to_move == WHITE && white_check) {
        return true;
    } else if (side_to_move == BLACK && black_check) {
        return true;
    }

    // Check if the side to move has no legal moves

    if (legal_moves(side_to_move).empty()) {
        return true;
    }

    return false;
}

void Game::check_control() {
    // Clear previous check status
    white_check = black_check = false;

    // Find the kings
    Square white_king_sq = NO_SQUARE, black_king_sq = NO_SQUARE;
    for (int sq = 0; sq < 64; ++sq) {
        auto piece_info = board.get_piece_on_square(static_cast<Square>(sq));
        if (piece_info.second == KING) {
            if (piece_info.first == WHITE)
                white_king_sq = static_cast<Square>(sq);
            else
                black_king_sq = static_cast<Square>(sq);
        }
    }

    // Check if the kings are under attack
    for (int sq = 0; sq < 64; ++sq) {
        auto piece_info = board.get_piece_on_square(static_cast<Square>(sq));
        if (piece_info.first == NO_COLOR)
            continue;

        auto moves = pseudo_legal_moves(static_cast<Square>(sq));
        for (auto move : moves) {
            if (move.to == white_king_sq)
                white_check = true;
            if (move.to == black_king_sq)
                black_check = true;
        }
    }
}


// Generate all legal moves for a given color
std::vector<Move> Game::legal_moves(Color color) const {
    std::vector<Move> moves;
    for (int sq = 0; sq < 64; ++sq) {
        auto piece_info = board.get_piece_on_square(static_cast<Square>(sq));
        if (piece_info.first == color) {
            auto piece_moves = legal_moves(static_cast<Square>(sq));
            moves.insert(moves.end(), piece_moves.begin(), piece_moves.end());
        }
    }
    return moves;
}

bool Game::is_move_legal(Move input_move) const {
    if (input_move.color == NO_COLOR)
        return false;

    // Optionally enforce moving only the side to move:
    if (input_move.color != side_to_move)
        return false;

    // First: must be a pseudo-legal move.
    auto moves = pseudo_legal_moves(input_move.from);
    bool found = false;
    for (auto target : moves) {
        if (target.to == input_move.to) {
            found = true;
            break;
        }
    }
    if (!found)
        return false;

    // ? should it be a copy of the Game object ?
    // Second: simulate the move and check king safety.
//     ChessBoard board_copy = board;
//     board_copy.do_move(input_move);
//     board_copy.check_control();
//     bool next_white_check = board_copy.get_check(WHITE);
//     bool next_black_check = board_copy.get_check(BLACK);

//     // The move is legal if it does not leave the mover's king in check.
//     if (side_to_move == WHITE)
//         return !next_white_check;
//     else
//         return !next_black_check;
    


    return true;
}

std::vector<Move> Game::pseudo_legal_moves(Square from) const {
    std::vector<Move> moves{};

    auto piece_info = board.get_piece_on_square(from);
    
    // If the square is empty, return an empty set
    if (piece_info.first == NO_COLOR)
        return {};

    Color mover = piece_info.first;
    Piece p = piece_info.second;
    int from_row, from_col;
    board.square_to_coord(from, from_row, from_col);

    // Loop over all destination squares
    for (int to = 0; to < 64; ++to) {
        if (to == from)
            continue;
        int to_row, to_col;
        board.square_to_coord(static_cast<Square>(to), to_row, to_col);
        int dr = to_row - from_row;
        int dc = to_col - from_col;

        // Skip if target square contains a friendly piece
        auto target_info = board.get_piece_on_square(static_cast<Square>(to));
        if (target_info.first == mover)
            continue;

        bool valid = false;
        switch (p) {
            case PAWN:
                if (mover == WHITE) {
                    // Single forward move
                    if (dc == 0 && dr == 1 && !board.is_occupied(static_cast<Square>(to)))
                        valid = true;
                    // Double move from starting rank 2
                    else if (dc == 0 && dr == 2 && from_row == 1 &&
                             !board.is_occupied(static_cast<Square>(to)) &&
                             !board.is_occupied(static_cast<Square>(from + 8)))
                        valid = true;
                    // Diagonal capture
                    else if (std::abs(dc) == 1 && dr == 1 && target_info.first == BLACK)
                        valid = true;
                    // en_passant capture
                    if (en_passant_square != NO_SQUARE) {
                        // if it's exactly one diagonal step from 'from'
                        int from_row = from / 8, from_col = from % 8;
                        int eps_row = en_passant_square / 8, eps_col = en_passant_square % 8;
                        if ((eps_row == from_row + 1) && (std::abs(eps_col - from_col) == 1)) {
                            Move move(mover, p, from, en_passant_square, EN_PASSANT);
                            moves.push_back(move);
                        }
                    }
                } else { // BLACK pawn
                    if (dc == 0 && dr == -1 && !board.is_occupied(static_cast<Square>(to)))
                        valid = true;
                    else if (dc == 0 && dr == -2 && from_row == 6 &&
                             !board.is_occupied(static_cast<Square>(to)) &&
                             !board.is_occupied(static_cast<Square>(from - 8)))
                        valid = true;
                    else if (std::abs(dc) == 1 && dr == -1 && target_info.first == WHITE)
                        valid = true;
                    // en_passant capture
                    if (en_passant_square != NO_SQUARE) {
                        // if it's exactly one diagonal step from 'from'
                        int from_row = from / 8, from_col = from % 8;
                        int eps_row = en_passant_square / 8, eps_col = en_passant_square % 8;
                        if ((eps_row == from_row - 1) && (std::abs(eps_col - from_col) == 1)) {
                            Move move{mover, p, from, en_passant_square, EN_PASSANT};
                            moves.push_back(move);
                        }
                    }
                }
                break;

            case KNIGHT:
                if ((std::abs(dr) == 2 && std::abs(dc) == 1) ||
                    (std::abs(dr) == 1 && std::abs(dc) == 2))
                    valid = true;
                break;

            case BISHOP:
                if (std::abs(dr) == std::abs(dc))
                    valid = board.is_path_clear(from, static_cast<Square>(to));
                break;

            case ROOK:
                if (dr == 0 || dc == 0)
                    valid = board.is_path_clear(from, static_cast<Square>(to));
                break;

            case QUEEN:
                if ((std::abs(dr) == std::abs(dc)) || (dr == 0 || dc == 0))
                    valid = board.is_path_clear(from, static_cast<Square>(to));
                break;

            case KING:
                if (std::abs(dr) <= 1 && std::abs(dc) <= 1)
                    valid = true;
                // Castling (only if king's normal one-square moves aren’t used)
                // Kingside castling for WHITE, for example:
                if (mover == WHITE && castling_rights[WHITE][1]) { // kingside right
                    // For white, squares F1 and G1 must be empty and not attacked.
                    if (!board.is_occupied(F1) && !board.is_occupied(G1) &&
                        board.is_path_clear(E1, G1)) { 
                        // Optionally: also ensure E1, F1, G1 are not under enemy attack.
                        // Add target square G1
                        Move move{mover, p, from, G1, CASTLING};
                        moves.push_back(move);
                    }
                } 
                if (mover == WHITE && castling_rights[WHITE][0]) { // queenside left
                    // For white, squares B1, C1, D1 must be empty and not attacked.
                    if (!board.is_occupied(B1) && !board.is_occupied(C1) && !board.is_occupied(D1) &&
                        board.is_path_clear(E1, C1)) {
                        // Optionally: also ensure E1, D1, C1 are not under enemy attack.
                        // Add target square C1
                        Move move{mover, p, from, C1, CASTLING};
                        moves.push_back(move);
                    }
                }
                if (mover == BLACK && castling_rights[BLACK][1]) { // kingside right
                    // For black, squares F8 and G8 must be empty and not attacked.
                    if (!board.is_occupied(F8) && !board.is_occupied(G8) &&
                        board.is_path_clear(E8, G8)) {
                        // Optionally: also ensure E8, F8, G8 are not under enemy attack.
                        // Add target square G8
                        Move move{mover, p, from, G8, CASTLING};
                        moves.push_back(move);
                    }
                }
                if (mover == BLACK && castling_rights[BLACK][0]) { // queenside left
                    // For black, squares B8, C8, D8 must be empty and not attacked.
                    if (!board.is_occupied(B8) && !board.is_occupied(C8) && !board.is_occupied(D8) &&
                        board.is_path_clear(E8, C8)) {
                        // Optionally: also ensure E8, D8, C8 are not under enemy attack.
                        // Add target square C8
                        Move move{mover, p, from, C8, CASTLING};
                        moves.push_back(move);
                    }
                }
                break;

            default:
                break;
        }
        if (valid)
            moves.push_back({mover, p, from, static_cast<Square>(to), NORMAL});
    }
    return moves;
}

std::vector<Move> Game::legal_moves(Square from) const {
    std::vector<Move> moves{};
    auto piece_info = board.get_piece_on_square(from);
    if (piece_info.first == NO_COLOR)
        return moves;

    for (auto move : pseudo_legal_moves(from)) {
        if (is_move_legal(move))
            moves.push_back(move);
    }
    return moves;
}



void Game::do_move(const Move& move) {
    Square from = move.from;
    Square to = move.to;
    Piece p = move.piece;
    MoveType type = move.type;

    // std::cout << "Applying move: " << square_to_string(move.from) << " -> " 
    //           << square_to_string(move.to) << std::endl;
    
    //           print();

    // Ensure the move is legal
    if (p == NO_PIECE) {
        std::cerr << "Error: No piece on the selected square!\n";
        return;
    }

    // Capture handling (if any)
    if (board.is_occupied(to) && type != CASTLING) {
        board.remove_piece(to);
    }

    // Handle en passant
    if (type == EN_PASSANT) {
        Square captured_pawn_square = static_cast<Square>(to + (side_to_move == WHITE ? -8 : 8));
        board.remove_piece(captured_pawn_square);
    }

    // Reset en passant square
    en_passant_square = NO_SQUARE;

    // Handle pawn-specific logic
    if (p == PAWN) {
        int from_row = from / 8;
        int to_row = to / 8;

        // Handle double pawn push
        if (type == DOUBLE_PAWN_PUSH) {
            en_passant_square = static_cast<Square>(((from_row + to_row) / 2) * 8 + (from % 8));
        }

        // Handle promotion
        if (type == PROMOTION) {
            p = choose_promotion_piece();
            // TODO: move.promoted_to = p;
        }
    }

    // Handle castling
    if (type == CASTLING) {
        if (to == G1) { board.remove_piece(H1); board.add_piece(WHITE, ROOK, F1); }
        if (to == C1) { board.remove_piece(A1); board.add_piece(WHITE, ROOK, D1); }
        if (to == G8) { board.remove_piece(H8); board.add_piece(BLACK, ROOK, F8); }
        if (to == C8) { board.remove_piece(A8); board.add_piece(BLACK, ROOK, D8); }
    }

    // Update castling rights if a rook moves or is captured
    if (p == ROOK || (board.is_occupied(to) && board.get_piece_on_square(to).second == ROOK)) {
        if (from == A1 || to == A1) castling_rights[WHITE][0] = false;
        if (from == H1 || to == H1) castling_rights[WHITE][1] = false;
        if (from == A8 || to == A8) castling_rights[BLACK][0] = false;
        if (from == H8 || to == H8) castling_rights[BLACK][1] = false;
    }

    // Move the piece
    board.remove_piece(from);
    board.add_piece(side_to_move, p, to);

    // Switch turns
    side_to_move = (side_to_move == WHITE) ? BLACK : WHITE;

    // std::cout << "Board after move:\n";
    // print();
}


void Game::undo_move(const Move& move) {
    // Get the moved piece and its color
    Color mover = (side_to_move == WHITE) ? BLACK : WHITE; // Undoing move, so switch turn back
    Piece p = move.piece;

    // Undo promotion: If the piece was promoted, revert it back to a pawn
    if (move.type == PROMOTION) {
        board.remove_piece(move.to);  // Remove promoted piece
        board.add_piece(mover, PAWN, move.from);  // Restore pawn
    } else {
        board.remove_piece(move.to);  // Remove moved piece
        board.add_piece(mover, p, move.from);
    }

    // Restore captured piece (if there was one)
    if (move.captured_piece != NO_PIECE) {
        Color captured_color = (mover == WHITE) ? BLACK : WHITE;
        board.add_piece(captured_color, move.captured_piece, move.to);
    }

    // Undo castling
    if (move.type == CASTLING) {
        if (mover == WHITE) {
            if (move.to == G1) { // Kingside
                board.remove_piece(F1);
                board.add_piece(WHITE, ROOK, H1);
            } else if (move.to == C1) { // Queenside
                board.remove_piece(D1);
                board.add_piece(WHITE, ROOK, A1);
            }
        } else if (mover == BLACK) {
            if (move.to == G8) { // Kingside
                board.remove_piece(F8);
                board.add_piece(BLACK, ROOK, H8);
            } else if (move.to == C8) { // Queenside
                board.remove_piece(D8);
                board.add_piece(BLACK, ROOK, A8);
            }
        }
        // Restore king position
        board.add_piece(mover, KING, move.from);
    }

    // Undo en passant
    if (move.type == EN_PASSANT) {
        if (mover == WHITE) {
            board.add_piece(BLACK, PAWN, static_cast<Square>(move.to - 8)); // Restore Black pawn
        } else {
            board.add_piece(WHITE, PAWN, static_cast<Square>(move.to + 8)); // Restore White pawn
        }
    }

    // Switch turns back
    side_to_move = mover;
}

std::pair<Square, Square> Game::parse_input(const std::string& from, const std::string& to) const {
    Square from_sq = static_cast<Square>(8 * (from[1] - '1') + (from[0] - 'a'));
    Square to_sq = static_cast<Square>(8 * (to[1] - '1') + (to[0] - 'a'));

    return {from_sq, to_sq};
}

Move Game::parse_move(const std::string& from, const std::string& to) const {
    Square from_sq = static_cast<Square>(8 * (from[1] - '1') + (from[0] - 'a'));
    Square to_sq = static_cast<Square>(8 * (to[1] - '1') + (to[0] - 'a'));

    // Get the moving piece
    Color color = board.get_piece_on_square(from_sq).first;
    Piece piece = board.get_piece_on_square(from_sq).second;
    
    Move move(color, piece, from_sq, to_sq, *this);
    
    return move;
}

void Game::play() {
    
    while (true) {

        turn += 1;
        board.print();

        // Check if the king is in check
        check_control();
        if(get_check(WHITE)) {
            std::cout << "\033[1;31mWhite in check!\033[0m" << std::endl;
        } else if(get_check(BLACK)) {
            std::cout << "\033[1;31mBlack in check!\033[0m" << std::endl;
        }

        // Checkmate
        if(legal_moves(get_side_to_move()).size() == 0 && get_check(get_side_to_move())) {
            Color color = get_side_to_move() == WHITE ? BLACK : WHITE;
            std::cout << "\033[1;31mCheckmate!\033[0m" << color << " wins!" << std::endl;
            break;
        }

        // printing color to move in blue and relative turn
        if(get_side_to_move() == WHITE) {
            std::cout << "\033[1;34mWhite to move (\033[0m" << (turn / 2) + 1 << "\033[1;34m)\033[0m" << std::endl;
        } else {
            std::cout << "\033[1;34mBlack to move (\033[0m" << (turn / 2) << "\033[1;34m)\033[0m" << std::endl;
        }
        
        std::string str_from;
        std::cout << "Enter piece to move (only its square): ";
        std::cin >> str_from;
        if(str_from == "exit") {
            break;
        }

        Square from = static_cast<Square>(8 * (str_from[1] - '1') + (str_from[0] - 'a'));

        // printing legal moves
        std::vector<Move> moves = legal_moves(from);
        std::cout << "Legal moves: ";
        for (const auto& move : moves) {
            std::cout << square_to_string(move.to) << " ";
        }
        std::cout << std::endl;

        std::cout << "Enter move (only its square): ";
        std::string str_to;
        std::cin >> str_to;
        if(str_to == "exit") {
            break;
        }    

        Move move = parse_move(str_from, str_to);

        // Check if the move is legal
        if (!is_move_legal(move)) {
            std::cout << "\033[1;31mIllegal move\033[0m\n";
            continue;
        }
        
        move.print();
        // Make the move
        do_move(move);
        move.print();
        
    }

}


// void Game::play_vs_pc(const int search_depth) {
//     while(true) {
//         turn += 1;
//         board.print();
//         board.check_control();
//         if(board.get_check(WHITE)) {
//             std::cout << "\033[1;31mWhite in check!\033[0m" << std::endl;
//         } else if(board.get_check(BLACK)) {
//             std::cout << "\033[1;31mBlack in check!\033[0m" << std::endl;
//         }

//         if(board.legal_moves(board.get_side_to_move()).size() == 0 && board.get_check(board.get_side_to_move())) {
//             Color color = board.get_side_to_move() == WHITE ? BLACK : WHITE;
//             std::cout << "\033[1;31mCheckmate!\033[0m" << color << " wins!" << std::endl;
//             break;
//         }

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
//         std::vector<Move> moves = board.legal_moves(from);
//         std::cout << "Legal moves: ";

//         for (const auto& move : moves) {
//             std::cout << square_to_string(move.to) << " ";
//         }

//         std::cout << std::endl;
        
//         std::cout << "Enter move (only its square): ";
//         std::cin >> inputMove;
//         if(inputMove == "exit") {
//             break;
//         }

//         // ? std::pair<Square, Square> move = parse_input(inputPiece, inputMove);

//         Move move = parse_move(inputPiece, inputMove);

//         // Check if the move is legal
//         if (!board.is_move_legal(move)) {
//             std::cout << "\033[1;31mIllegal move\033[0m\n";
//             continue;
//         }

//         // Make the move
//         board.do_move(move);

//         // Evaluate the position
//         int score = evaluate(board);
//         std::cout << "Position score: " << score << std::endl;

//         std::cout << "Looking for best move...\n";

//         // Measure time taken to find best move & find best move
//         auto start = std::chrono::high_resolution_clock::now();
//         Move best_move = find_best_move(board, search_depth);
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsed = end - start;
//         std::cout << "Time taken to find best move: " << elapsed.count() << " seconds" << std::endl;
        
//         // Print best move
//         std::cout << "Best move: " << square_to_string(best_move.from) 
//                   << " to " << square_to_string(best_move.to) << std::endl;

//     }
// }

