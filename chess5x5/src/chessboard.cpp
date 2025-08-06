#include "chessboard.hpp"
#include "bitboard.hpp"
#include <iostream>


void ChessBoard::reset() {
    // Clear all pieces
    ChessBoard::clear();

    // Endgame position with 4 pieces: Black King, White Queen, White Rook, White King
    // Let's place them in a meaningful endgame position on 5x5 board
    // Black King on A5 (square 20)
    // White Queen on C3 (square 12) 
    // White Rook on E2 (square 9)
    // White King on E1 (square 4)
    
    pieces[BLACK][KING]   = 1U << 20;  // A5
    pieces[WHITE][QUEEN]  = 1U << 12;  // C3
    pieces[WHITE][ROOK]   = 1U << 9;   // E2
    pieces[WHITE][KING]   = 1U << 4;   // E1
}

// Print board (for debugging)
void ChessBoard::print() const {
    const char piece_symbols[2][6] = {
        {'P', 'N', 'B', 'R', 'Q', 'K'}, // White pieces
        {'p', 'n', 'b', 'r', 'q', 'k'}  // Black pieces
    };

    for (int rank = 4; rank >= 0; rank--) {
        std::cout << (rank + 1) << " ";
        for (int file = 0; file < 5; file++) {
            int sq = rank * 5 + file;
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

    std::cout << "  a b c d e\n\n";
}

/* --------------------------------- Getter -------------------------------- */

// Get all pieces of a type/color
U32 ChessBoard::get_pieces(Color color, Piece piece) const {
    return pieces[color][piece];
}

std::vector<std::vector<std::string>> ChessBoard::get_board() const {
    std::vector<std::vector<std::string>> board(5, std::vector<std::string>(5, " "));
    for (int rank = 0; rank < 5; rank++) {
        for (int file = 0; file < 5; file++) {
            int sq = rank * 5 + file;
            for (int color = 0; color < 2; color++) {
                for (int piece = 0; piece < 6; piece++) {
                    if (Bitboard::get_bit(pieces[color][piece], static_cast<Square>(sq))) {
                        board[rank][file] = (color == WHITE ? "w" : "b") + std::to_string(piece);
                    }
                }
            }
        }
    }
    return board;
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


/* --------------------------------- Setter --------------------------------- */

// add a piece
void ChessBoard::add_piece(Color color, Piece piece, Square sq) {
    Bitboard::set_bit(pieces[color][piece], sq);
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

/* --------------------------------- Utility -------------------------------- */

// Check if the path between two squares is clear (excluding the destination)
bool ChessBoard::is_path_clear(Square from, Square to) const {
    int from_row, from_col, to_row, to_col;
    square_to_coord(from, from_row, from_col);
    square_to_coord(to, to_row, to_col);

    int dr = to_row - from_row;
    int dc = to_col - from_col;
    int step_r = (dr == 0 ? 0 : (dr > 0 ? 1 : -1));
    int step_c = (dc == 0 ? 0 : (dc > 0 ? 1 : -1));

    // Move from the square next to 'from' until reaching 'to'
    int r = from_row + step_r;
    int c = from_col + step_c;
    while (r != to_row || c != to_col) {
        if (is_occupied(static_cast<Square>(r * 5 + c)))
            return false;
        r += step_r;
        c += step_c;
    }
    return true;
}

// Check if any piece occupies a square
bool ChessBoard::is_occupied(Square sq) const {
    U32 all_pieces = 0;
    for (const auto& color_pieces : pieces) {
        for (const auto& piece_board : color_pieces) {
            all_pieces |= piece_board;
        }
    }
    return Bitboard::get_bit(all_pieces, sq);
}