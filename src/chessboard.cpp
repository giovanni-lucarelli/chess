// chessboard.cpp
#include "chessboard.hpp"
#include "bitboard.hpp"
#include <iostream>


// Constructor
ChessBoard::ChessBoard() {
    reset();
}

// Copy constructor
ChessBoard::ChessBoard(const ChessBoard& other) {
    pieces = other.pieces;
    side_to_move = other.side_to_move;
    en_passant_square = other.en_passant_square;
    castling_rights[WHITE][0] = other.castling_rights[WHITE][0];
    castling_rights[WHITE][1] = other.castling_rights[WHITE][1];
    castling_rights[BLACK][0] = other.castling_rights[BLACK][0];
    castling_rights[BLACK][1] = other.castling_rights[BLACK][1];
    white_check = other.white_check;
    black_check = other.black_check;
    // halfmove_clock = other.halfmove_clock;
    // fullmove_number = other.fullmove_number;
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
    auto piece_info = get_piece_on_square(from);
    Color mover = piece_info.first;
    Piece p = piece_info.second;

    // Remove captured piece if destination is occupied (or en passant, etc.)
    if (is_occupied(to)) {
        remove_piece(to);
    }
    if (p == PAWN && to == en_passant_square) {
        // Assuming eps_capture_sq was determined earlier
        remove_piece(static_cast<Square>(to - 8 * (mover == WHITE ? 1 : -1)));
    }

    // Clear en passant square by default.
    en_passant_square = NO_SQUARE;

    // Check for pawn moving two squares to set en passant.
    if (p == PAWN) {
        int from_row = static_cast<int>(from) / 8;
        int to_row = static_cast<int>(to) / 8;
        // Update en passant for two-square advance.
        if (std::abs(from_row - to_row) == 2) {
            en_passant_square = static_cast<Square>(((from_row + to_row) / 2) * 8 + (from % 8));
        }
        // Promotion condition: white pawn reaching rank 8; black pawn reaching rank 1.
        // Here rows are 0-indexed: row 7 for white and row 0 for black.
        if ((mover == WHITE && to_row == 7) || (mover == BLACK && to_row == 0)) {
            // You can call a helper here, for example, ask user input or default to QUEEN.
            // For example, default promotion to queen:
            p = QUEEN;
            // Alternatively, implement a function like choose_promotion_piece() to decide.
        }
    }
    
    // Remove pawn from initial square and add it (or its replacement) to the destination.
    remove_piece(from);
    add_piece(mover, p, to);

    // Switch turns
    side_to_move = (side_to_move == WHITE) ? BLACK : WHITE;
}

// A helper: convert a square index to row and column.
inline void square_to_coord(Square sq, int &row, int &col) {
    row = sq / 8;
    col = sq % 8;
}

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

// Generate pseudo-legal target squares for the piece on 'from'.
// (These moves follow the piece's movement rules but do not check king safety.)
std::set<Square> ChessBoard::pseudo_legal_targets(Square from) const {
    std::vector<Square> targets;
    auto piece_info = get_piece_on_square(from);
    
    // If the square is empty, return an empty set
    if (piece_info.first == NO_COLOR)
        return {};

    Color mover = piece_info.first;
    Piece p = piece_info.second;
    int from_row, from_col;
    square_to_coord(from, from_row, from_col);

    // Loop over all destination squares
    for (int to = 0; to < 64; ++to) {
        if (to == from)
            continue;
        int to_row, to_col;
        square_to_coord(static_cast<Square>(to), to_row, to_col);
        int dr = to_row - from_row;
        int dc = to_col - from_col;

        // Skip if target square contains a friendly piece
        auto target_info = get_piece_on_square(static_cast<Square>(to));
        if (target_info.first == mover)
            continue;

        bool valid = false;
        switch (p) {
            case PAWN:
                if (mover == WHITE) {
                    // Single forward move
                    if (dc == 0 && dr == 1 && !is_occupied(static_cast<Square>(to)))
                        valid = true;
                    // Double move from starting rank 2
                    else if (dc == 0 && dr == 2 && from_row == 1 &&
                             !is_occupied(static_cast<Square>(to)) &&
                             !is_occupied(static_cast<Square>(from + 8)))
                        valid = true;
                    // Diagonal capture
                    else if (std::abs(dc) == 1 && dr == 1 && target_info.first == BLACK)
                        valid = true;
                    // en passant capture
                    if (en_passant_square != NO_SQUARE) {
                        // if it's exactly one diagonal step from 'from'
                        int from_row = from / 8, from_col = from % 8;
                        int eps_row = en_passant_square / 8, eps_col = en_passant_square % 8;
                        if ((eps_row == from_row + 1) && (std::abs(eps_col - from_col) == 1)) {
                            targets.push_back(en_passant_square);
                        }
                    }
                } else { // BLACK pawn
                    if (dc == 0 && dr == -1 && !is_occupied(static_cast<Square>(to)))
                        valid = true;
                    else if (dc == 0 && dr == -2 && from_row == 6 &&
                             !is_occupied(static_cast<Square>(to)) &&
                             !is_occupied(static_cast<Square>(from - 8)))
                        valid = true;
                    else if (std::abs(dc) == 1 && dr == -1 && target_info.first == WHITE)
                        valid = true;
                    // en passant capture
                    if (en_passant_square != NO_SQUARE) {
                        // if it's exactly one diagonal step from 'from'
                        int from_row = from / 8, from_col = from % 8;
                        int eps_row = en_passant_square / 8, eps_col = en_passant_square % 8;
                        if ((eps_row == from_row - 1) && (std::abs(eps_col - from_col) == 1)) {
                            targets.push_back(en_passant_square);
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
                    valid = is_path_clear(from, static_cast<Square>(to));
                break;

            case ROOK:
                if (dr == 0 || dc == 0)
                    valid = is_path_clear(from, static_cast<Square>(to));
                break;

            case QUEEN:
                if ((std::abs(dr) == std::abs(dc)) || (dr == 0 || dc == 0))
                    valid = is_path_clear(from, static_cast<Square>(to));
                break;

            case KING:
                if (std::abs(dr) <= 1 && std::abs(dc) <= 1)
                    valid = true;
                // (Optional: add castling rules here)
                break;

            default:
                break;
        }
        if (valid)
            targets.push_back(static_cast<Square>(to));
    }
    return std::set<Square>(targets.begin(), targets.end());
}


// Controlling check conditions for both colors
void ChessBoard::check_control() {
    // Clear previous check status
    white_check = black_check = false;

    // Find the kings
    Square white_king_sq = NO_SQUARE, black_king_sq = NO_SQUARE;
    for (int sq = 0; sq < 64; ++sq) {
        auto piece_info = get_piece_on_square(static_cast<Square>(sq));
        if (piece_info.second == KING) {
            if (piece_info.first == WHITE)
                white_king_sq = static_cast<Square>(sq);
            else
                black_king_sq = static_cast<Square>(sq);
        }
    }

    // Check if the kings are under attack
    for (int sq = 0; sq < 64; ++sq) {
        auto piece_info = get_piece_on_square(static_cast<Square>(sq));
        if (piece_info.first == NO_COLOR)
            continue;

        auto targets = pseudo_legal_targets(static_cast<Square>(sq));
        for (Square target : targets) {
            if (target == white_king_sq)
                white_check = true;
            if (target == black_king_sq)
                black_check = true;
        }
    }
}



// Check if a move from 'from' to 'to' is fully legal.
// First we check that it is pseudo-legal, then we simulate the move and ensure that it
// does not leave the mover’s king in check.
bool ChessBoard::is_move_legal(Square from, Square to) const {
    auto piece_info = get_piece_on_square(from);
    if (piece_info.first == NO_COLOR)
        return false;

    // Optionally enforce moving only the side to move:
    if (piece_info.first != side_to_move)
        return false;

    // First: must be a pseudo-legal move.
    auto targets = pseudo_legal_targets(from);
    bool found = false;
    for (Square target : targets) {
        if (target == to) {
            found = true;
            break;
        }
    }
    if (!found)
        return false;


    // Second: simulate the move and check king safety.
    ChessBoard board_copy = *this;
    board_copy.move_piece(from, to);
    board_copy.check_control();
    bool next_white_check = board_copy.get_check(WHITE);
    bool next_black_check = board_copy.get_check(BLACK);

    // The move is legal if it does not leave the mover's king in check.
    if (side_to_move == WHITE)
        return !next_white_check;
    else
        return !next_black_check;
    


    return true;
}



// Generate all legal moves for the piece on square 'from', remember that it cannot leave the king in check the next state
std::vector<std::pair<Square, Square>> ChessBoard::legal_moves(Square from) const {
    std::vector<std::pair<Square, Square>> moves;
    auto piece_info = get_piece_on_square(from);
    if (piece_info.first == NO_COLOR)
        return moves;

    auto targets = pseudo_legal_targets(from);
    for (Square target : targets) {
        if (is_move_legal(from, target))
            moves.push_back({from, target});
    }
    return moves;
}

// Generate all legal moves for a given color
std::vector<std::pair<Square, Square>> ChessBoard::legal_moves(Color color) const {
    std::vector<std::pair<Square, Square>> moves;
    for (int sq = 0; sq < 64; ++sq) {
        auto piece_info = get_piece_on_square(static_cast<Square>(sq));
        if (piece_info.first == color) {
            auto piece_moves = legal_moves(static_cast<Square>(sq));
            moves.insert(moves.end(), piece_moves.begin(), piece_moves.end());
        }
    }
    return moves;
}

Piece ChessBoard::choose_promotion_piece() const {
    std::cout << "Promote pawn to (q, r, b, n): ";
    char choice;
    std::cin >> choice;
    switch(tolower(choice)) {
        case 'q': return QUEEN;
        case 'r': return ROOK;
        case 'b': return BISHOP;
        case 'n': return KNIGHT;
        default: return QUEEN; // default promotion to queen
    }
}

