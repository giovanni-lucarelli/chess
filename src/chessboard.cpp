#include "chessboard.hpp"
#include "bitboard.hpp"
#include <iostream>


void ChessBoard::reset() {
    // Clear all pieces
    ChessBoard::clear();

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

    
    // halfmove_clock = 0;
    // fullmove_number = 1;
}

// Print board (for debugging)
// void ChessBoard::print() const {
//     const char piece_symbols[2][6] = {
//         {'P', 'N', 'B', 'R', 'Q', 'K'}, // White pieces
//         {'p', 'n', 'b', 'r', 'q', 'k'}  // Black pieces
//     };

//     for (int rank = 7; rank >= 0; rank--) {
//         std::cout << (rank + 1) << " ";
//         for (int file = 0; file < 8; file++) {
//             int sq = rank * 8 + file;
//             char piece_char = ' ';
//             for (int color = 0; color < 2; color++) {
//                 for (int piece = 0; piece < 6; piece++) {
//                     if (Bitboard::get_bit(pieces[color][piece], static_cast<Square>(sq))) {
//                         piece_char = piece_symbols[color][piece];
//                     }
//                 }
//             }
//             std::cout << piece_char << " ";
//         }
//         std::cout << "\n";
//     }

//     std::cout << "  a b c d e f g h\n\n";
// }
// write on a string instead of printing to console

std::string ChessBoard::print() const {
    std::string output;
    const char piece_symbols[2][6] = {
        {'P', 'N', 'B', 'R', 'Q', 'K'}, // White pieces
        {'p', 'n', 'b', 'r', 'q', 'k'}  // Black pieces
    };

    for (int rank = 7; rank >= 0; rank--) {
        output += std::to_string(rank + 1) + " ";
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
            output += piece_char;
            output += " ";
        }
        output += "\n";
    }

    output += "  a b c d e f g h\n\n";
    return output;
}

/* --------------------------------- Getter -------------------------------- */

// Get all pieces of a type/color
U64 ChessBoard::get_pieces(Color color, Piece piece) const {
    return pieces[color][piece];
}

std::vector<std::vector<std::string>> ChessBoard::get_board() const {
    std::vector<std::vector<std::string>> board(8, std::vector<std::string>(8, " "));
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
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
bool ChessBoard::add_piece(Color color, Piece piece, Square sq) {
    Bitboard::set_bit(pieces[color][piece], sq);
    return true;
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
        if (is_occupied(static_cast<Square>(r * 8 + c)))
            return false;
        r += step_r;
        c += step_c;
    }
    return true;
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