// chessboard.cpp
#include "chessboard.hpp"
#include "bitboard.hpp"
#include "movegen.hpp"
#include <iostream>

ChessBoard::ChessBoard() : en_passant_square(H8) { // H8 = no en passant
    this->reset();
}

void ChessBoard::reset() {
    // Clear all pieces
    for (auto& color_pieces : pieces) {
        color_pieces.fill(0);
    }

    // White pieces
    pieces[WHITE][PAWN]   = 0x000000000000FF00ULL; // Rank 2
    pieces[WHITE][KNIGHT] = 0x0000000000000042ULL; // B1, G1
    pieces[WHITE][BISHOP] = 0x0000000000000024ULL; // C1, F1
    pieces[WHITE][ROOK]   = 0x0000000000000081ULL; // A1, H1
    pieces[WHITE][QUEEN]  = 0x0000000000000008ULL; // D1
    pieces[WHITE][KING]   = 0x0000000000000010ULL; // E1

    // Black pieces
    pieces[BLACK][PAWN]   = 0x00FF000000000000ULL; // Rank 7
    pieces[BLACK][KNIGHT] = 0x4200000000000000ULL; // B8, G8
    pieces[BLACK][BISHOP] = 0x2400000000000000ULL; // C8, F8
    pieces[BLACK][ROOK]   = 0x8100000000000000ULL; // A8, H8
    pieces[BLACK][QUEEN]  = 0x0800000000000000ULL; // D8
    pieces[BLACK][KING]   = 0x1000000000000000ULL; // E8

    // Game state
    side_to_move = WHITE;
    castling_rights[WHITE][0] = castling_rights[WHITE][1] = true; // QK
    castling_rights[BLACK][0] = castling_rights[BLACK][1] = true;
    en_passant_square = H8; // No en passant
    // halfmove_clock = 0;
    // fullmove_number = 1;
}

// Check if any piece occupies a square
bool ChessBoard::is_occupied(Square sq) const {
    U64 all_pieces = 0;
    for (const auto& color_pieces : pieces) {
        for (const auto& piece_board : color_pieces) {
            all_pieces |= piece_board;
        }
    }
    return Bitboard::get_bit(all_pieces, sq);
}

// Get all pieces of a type/color
U64 ChessBoard::get_pieces(Color color, Piece piece) const {
    return pieces[color][piece];
}

// Print board (for debugging)
void ChessBoard::print() const {
    const char piece_symbols[2][6] = {
        {'P', 'N', 'B', 'R', 'Q', 'K'}, // White pieces
        {'p', 'n', 'b', 'r', 'q', 'k'}  // Black pieces
    };

    for (int rank = 7; rank >= 0; rank--) {
        std::cout << (rank + 1) << " ";
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            char piece_char = ' ';
            for (int color = 0; color < 2; color++) {
                for (int piece = 0; piece < 6; piece++) {
                    if (Bitboard::get_bit(pieces[color][piece], static_cast<Square>(sq))) {
                        piece_char = piece_symbols[color][piece];
                    }
                }
            }
            std::cout << piece_char << " ";
        }
        std::cout << "\n";
    }

    std::cout << "  a b c d e f g h\n\n";
}

// return the piece and the color on a square
std::pair<Color, Piece> ChessBoard::get_piece_on_square(Square sq) const {
    for (int color = 0; color < 2; color++) {
        for (int piece = 0; piece < 6; piece++) {
            if (Bitboard::get_bit(pieces[color][piece], sq)) {
                return {static_cast<Color>(color), static_cast<Piece>(piece)};
            }
        }
    }
    return {NO_COLOR, NO_PIECE};
}

// remove a piece
void ChessBoard::remove_piece(Square sq) {
    for (int color = 0; color < 2; color++) {
        for (int piece = 0; piece < 6; piece++) {
            if (Bitboard::get_bit(pieces[color][piece], sq)) {
                Bitboard::clear_bit(pieces[color][piece], sq);
                return;
            }
        }
    }
}

// add a piece
void ChessBoard::add_piece(Color color, Piece piece, Square sq) {
    Bitboard::set_bit(pieces[color][piece], sq);
}

// move a piece
void ChessBoard::move_piece(Square from, Square to) {
    // capture
    if (is_occupied(to)) {
        remove_piece(to);
    }
    
    std::pair<Color, Piece> piece = get_piece_on_square(from);
    remove_piece(from);
    add_piece(piece.first, piece.second, to);


    // checking if the opponent king is in check
    Square opponent_king_sq;
    Color opponent_color = (side_to_move == WHITE) ? BLACK : WHITE;
    for (int i = 0; i < 64; i++) {
        if (get_piece_on_square(static_cast<Square>(i)) == std::make_pair(opponent_color, KING)) {
            opponent_king_sq = static_cast<Square>(i);
        }
    }
    if(DEBUG) {
        std::cout << "\033[1;34mOpponent king square:\033[0m " << square_to_string(opponent_king_sq) << std::endl;
    }
    std::cout << "\033[1;32mNext legal moves:\033[0m" << std::endl;
    for(auto move : legal_moves(to, side_to_move)){
        std::cout << square_to_string(move.to) << std::endl;
        if(move.to == opponent_king_sq){
            std::cout << "\033[1;31mCheck!\033[0m" << std::endl;
            check = true;
        }
    }


    side_to_move = (side_to_move == WHITE) ? BLACK : WHITE;
    
}

// Check if the path is clear
bool ChessBoard::is_path_clear(Square from, Square to) const {
    // Horizontal check
    if (from / 8 == to / 8) {
        int start = std::min(from, to) + 1;
        int end = std::max(from, to);
        for (int sq = start; sq < end; sq++) {
            if (is_occupied(static_cast<Square>(sq))) {
                return false;
            }
        }
        return true;
    }

    // Vertical check
    if (from % 8 == to % 8) {
        int start = std::min(from, to) + 8;
        int end = std::max(from, to);
        for (int sq = start; sq < end; sq += 8) {
            if (is_occupied(static_cast<Square>(sq))) {
                return false;
            }
        }
        return true;
    }

    // Diagonal check
    if (std::abs(from - to) % 7 == 0 ) {
        int start = std::min(from, to) + 7;
        int end = std::max(from, to);
        for (int sq = start; sq < end; sq += 7) {
            if (is_occupied(static_cast<Square>(sq))) {
                return false;
            }
        }
        return true;
    }
    if (std::abs(from - to) % 9 == 0 ) {
        int start = std::min(from, to) + 9;
        int end = std::max(from, to);
        for (int sq = start; sq < end; sq += 9) {
            if (is_occupied(static_cast<Square>(sq))) {
                return false;
            }
        }
        return true;
    }
}


// Check if a move is legal
bool ChessBoard::is_move_legal(Square from, Square to) const {
    // chek if the turn is correct
    if (get_piece_on_square(from).first != side_to_move) {
        return false;
    }

    // Check if the from square is occupied
    if (!is_occupied(from)) {
        return false;
    }

    // Check if the to square is occupied by a friendly piece
    std::pair<Color, Piece> from_piece = get_piece_on_square(from);
    std::pair<Color, Piece> to_piece = get_piece_on_square(to);
    if (to_piece.first == from_piece.first) {
        return false;
    }
    
    int dx;
    int dy;

    

    // Check if the move is valid for the piece
    switch (from_piece.second) {
        case PAWN:
            // Checking if the pawn is in the starting row
            if(from_piece.first == WHITE && from / 8 == 1) {
                if(to - from == 8 || to - from == 16) {
                    // Also checking if in front of it is empty
                    return !is_occupied(to);
                }
            } else if(from_piece.first == BLACK && from / 8 == 6) {
                if(from - to == 8 || from - to == 16) {
                    return !is_occupied(to);
                }
            } else {
                if(from_piece.first == WHITE) {
                    if(to - from == 8) {
                        return !is_occupied(to);
                    }
                } else {
                    if(from - to == 8) {
                        return !is_occupied(to);
                    }
                }
            }
            // Capturing
            if(std::abs(from - to) == 7 || std::abs(from - to) == 9) {
                // Checking if the square is occupied for capturing
                return is_occupied(to);
            }
            break;
        case KNIGHT:
            // Check if the knight is moving in an L-shape
            dx = std::abs((to % 8) - (from % 8));
            dy = std::abs((to / 8) - (from / 8));
            if ((dx == 2 && dy == 1) || (dx == 1 && dy == 2)) {
                return true;
            } 
            break;
        case BISHOP:
            // Check if the bishop is moving diagonally
            if (std::abs(from - to) % 7 == 0 || std::abs(from - to) % 9 == 0) {
                return ChessBoard::is_path_clear(from, to);
            }
            break;
        case ROOK:
            // Check if the rook is moving horizontally or vertically
            if (from / 8 == to / 8 || from % 8 == to % 8) {
                // check if the path is clear
                return ChessBoard::is_path_clear(from, to);
            }
            break;
        case QUEEN:
            // Check if the queen is moving diagonally, horizontally, or vertically
            if (std::abs(from - to) % 7 == 0 || std::abs(from - to) % 9 == 0 ||
                from / 8 == to / 8 || from % 8 == to % 8) {
                return ChessBoard::is_path_clear(from, to);
            }
            break;
        case KING:
            // Check if the king is moving one square in any direction
            if (std::abs(from - to) == 1 || std::abs(from - to) == 7 ||
                std::abs(from - to) == 8 || std::abs(from - to) == 9) {
                return true;
            }
            break;
    }
    return false;
}

bool ChessBoard::is_in_check(Color color) const {
    // Searching for the king
    Square king_sq;
    for (int i = 0; i < 64; i++) {
        if (get_piece_on_square(static_cast<Square>(i)) == std::make_pair(color, KING)) {
            king_sq = static_cast<Square>(i);
        }
    }
    // Check 
}

// Check if the game is over
bool ChessBoard::is_checkmate() const {
    // Searching for the king
    Square king_sq;
    for (int i = 0; i < 64; i++) {
        if (get_piece_on_square(static_cast<Square>(i)) == std::make_pair(side_to_move, KING)) {
            king_sq = static_cast<Square>(i);
        }
    }
    // check if legal_moves is empty
    if (legal_moves(king_sq, WHITE).empty() || legal_moves(king_sq, BLACK).size() == 0) {
        return true;
    }
}

std::vector<Move> ChessBoard::legal_moves(Square piece, Color color) const {
    std::vector<Move> moves;
    for (int i = 0; i < 64; i++) {
        if (is_move_legal(piece, static_cast<Square>(i))) {
            moves.push_back(Move(piece, static_cast<Square>(i)));
        }
    }
    return moves;
}


