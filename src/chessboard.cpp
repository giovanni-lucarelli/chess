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
            char piece_char = '.';
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

// // check legality of a move
// bool ChessBoard::is_move_legal(Square from, Square to) const {
//     // check if from and to are in the proper range
//     if (from < A1 || from > H8 || to < A1 || to > H8) {
//         return false;
//     }

//     // Check if the move is valid
//     auto [color, piece] = get_piece_on_square(from);
//     if (color != side_to_move) {
//         return false;
//     }

//     // Check if the destination square is occupied by a friendly piece
//     if (is_occupied(to)) {
//         auto [dest_color, dest_piece] = get_piece_on_square(to);
//         if (dest_color == color) {
//             return false;
//         }
//     }

//     // Check if the move is valid for the piece
//     switch (piece) {
//         case PAWN:
//             // Check if the destination square is occupied by an enemy piece
//             if (is_occupied(to)) {
//                 auto [dest_color, dest_piece] = get_piece_on_square(to);
//                 if (dest_color != color) {
//                     return true;
//                 }
//             }
//             break;
//         case KNIGHT:
//             // only L-shaped moves
//             if (abs(from - to) != 6 && abs(from - to) != 10 && abs(from - to) != 15 && abs(from - to) != 17) {
//                 return false;
//             }
//             break;
//         case BISHOP:
//             // only diagonal moves
//             if (abs(from - to) % 9 != 0 && abs(from - to) % 7 != 0) {
//                 return false;
//             }
//             break;
//         case ROOK:
//             // only straight moves  
//             if (from / 8 != to / 8 && from % 8 != to % 8) {
//                 return false;
//             }
//             break;
//         case QUEEN:
//             // only diagonal or straight moves
//             if (abs(from - to) % 9 != 0 && abs(from - to) % 7 != 0 && (from / 8 != to / 8) && (from % 8 != to % 8)) {
//                 return false;
//             }
//             break;
//         case KING:
//             // only one square move
//             if (abs(from - to) != 1 && abs(from - to) != 8 && abs(from - to) != 9 && abs(from - to) != 7) {
//                 return false;
//             }
//             // TODO: Check for castling
//             // update castling status
//             // check three rules for castling
//             break;
//     }

//     return true;
// }

// move a piece
void ChessBoard::move_piece(Square from, Square to) {
    // Check if the move is legal

    if (!is_move_legal(from, to)) {
        throw std::runtime_error("Illegal move");
    }

    // Move the piece
    auto [color, piece] = get_piece_on_square(from);
    remove_piece(from);
    add_piece(color, piece, to);

    // Update game state
    side_to_move = static_cast<Color>(1 - color);
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


std::vector<Move> ChessBoard::generate_legal_moves() const {
    return MoveGenerator::generate_moves(*this);
}
