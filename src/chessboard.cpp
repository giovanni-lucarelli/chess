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

// A helper: convert a square index to row and column.
inline void square_to_coord(Square sq, int &row, int &col) {
    row = sq / 8;
    col = sq % 8;
}

Piece ChessBoard::choose_promotion_piece() const {
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


/* -------------------------------------------------------------------------- */
/*                             Checkers and Rules                             */
/* -------------------------------------------------------------------------- */

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

bool ChessBoard::is_game_over() {
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

// Controlling check conditions for both colors
// si può scrivere meglio usando il Move.type ?
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

        auto moves = pseudo_legal_moves(static_cast<Square>(sq));
        for (auto move : moves) {
            if (move.to == white_king_sq)
                white_check = true;
            if (move.to == black_king_sq)
                black_check = true;
        }
    }
}

// Generate pseudo-legal target squares for the piece on 'from'.
// (These moves follow the piece's movement rules but do not check king safety.)
std::vector<Move> ChessBoard::pseudo_legal_moves(Square from) const {
    std::vector<Move> moves{};

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
                    // en_passant capture
                    if (en_passant_square != NO_SQUARE) {
                        // if it's exactly one diagonal step from 'from'
                        int from_row = from / 8, from_col = from % 8;
                        int eps_row = en_passant_square / 8, eps_col = en_passant_square % 8;
                        if ((eps_row == from_row + 1) && (std::abs(eps_col - from_col) == 1)) {
                            Move move{mover, p, from, en_passant_square, EN_PASSANT};
                            moves.push_back(move);
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
                // Castling (only if king's normal one-square moves aren’t used)
                // Kingside castling for WHITE, for example:
                if (mover == WHITE && castling_rights[WHITE][1]) { // kingside right
                    // For white, squares F1 and G1 must be empty and not attacked.
                    if (!is_occupied(F1) && !is_occupied(G1) &&
                        is_path_clear(E1, G1)) { 
                        // Optionally: also ensure E1, F1, G1 are not under enemy attack.
                        // Add target square G1
                        Move move{mover, p, from, G1, CASTLING};
                        moves.push_back(move);
                    }
                } 
                if (mover == WHITE && castling_rights[WHITE][0]) { // queenside left
                    // For white, squares B1, C1, D1 must be empty and not attacked.
                    if (!is_occupied(B1) && !is_occupied(C1) && !is_occupied(D1) &&
                        is_path_clear(E1, C1)) {
                        // Optionally: also ensure E1, D1, C1 are not under enemy attack.
                        // Add target square C1
                        Move move{mover, p, from, C1, CASTLING};
                        moves.push_back(move);
                    }
                }
                if (mover == BLACK && castling_rights[BLACK][1]) { // kingside right
                    // For black, squares F8 and G8 must be empty and not attacked.
                    if (!is_occupied(F8) && !is_occupied(G8) &&
                        is_path_clear(E8, G8)) {
                        // Optionally: also ensure E8, F8, G8 are not under enemy attack.
                        // Add target square G8
                        Move move{mover, p, from, G8, CASTLING};
                        moves.push_back(move);
                    }
                }
                if (mover == BLACK && castling_rights[BLACK][0]) { // queenside left
                    // For black, squares B8, C8, D8 must be empty and not attacked.
                    if (!is_occupied(B8) && !is_occupied(C8) && !is_occupied(D8) &&
                        is_path_clear(E8, C8)) {
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

std::vector<Move> ChessBoard::legal_moves(Square from) const {
    std::vector<Move> moves{};
    auto piece_info = get_piece_on_square(from);
    if (piece_info.first == NO_COLOR)
        return moves;

    auto moves = pseudo_legal_moves(from);

    for (auto move : moves) {
        if (is_move_legal(move))
            moves.push_back(move);
    }
    return moves;
}

// Generate all legal moves for a given color
std::vector<Move> ChessBoard::legal_moves(Color color) const {
    std::vector<Move> moves;
    for (int sq = 0; sq < 64; ++sq) {
        auto piece_info = get_piece_on_square(static_cast<Square>(sq));
        if (piece_info.first == color) {
            auto piece_moves = legal_moves(static_cast<Square>(sq));
            moves.insert(moves.end(), piece_moves.begin(), piece_moves.end());
        }
    }
    return moves;
}

bool ChessBoard::is_move_legal(Move input_move) const {
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


    // Second: simulate the move and check king safety.
    ChessBoard board_copy = *this;
    board_copy.do_move(input_move);
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

/* -------------------------------------------------------------------------- */
/*                               Piece Movement                               */
/* -------------------------------------------------------------------------- */

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


void ChessBoard::do_move(const Move& move) {
    Square from = move.from;
    Square to = move.to;
    Piece p = move.piece;
    MoveType type = move.type;

    // Ensure the move is legal
    if (p == NO_PIECE) {
        std::cerr << "Error: No piece on the selected square!\n";
        return;
    }

    // Capture handling (if any)
    if (is_occupied(to) && type != CASTLING) {
        remove_piece(to);
    }

    // Handle en passant
    if (type == EN_PASSANT) {
        Square captured_pawn_square = static_cast<Square>(to + (side_to_move == WHITE ? -8 : 8));
        remove_piece(captured_pawn_square);
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
        if (to == G1) { remove_piece(H1); add_piece(WHITE, ROOK, F1); }
        if (to == C1) { remove_piece(A1); add_piece(WHITE, ROOK, D1); }
        if (to == G8) { remove_piece(H8); add_piece(BLACK, ROOK, F8); }
        if (to == C8) { remove_piece(A8); add_piece(BLACK, ROOK, D8); }
    }

    // Update castling rights if a rook moves or is captured
    if (p == ROOK || (is_occupied(to) && get_piece_on_square(to).second == ROOK)) {
        if (from == A1 || to == A1) castling_rights[WHITE][0] = false;
        if (from == H1 || to == H1) castling_rights[WHITE][1] = false;
        if (from == A8 || to == A8) castling_rights[BLACK][0] = false;
        if (from == H8 || to == H8) castling_rights[BLACK][1] = false;
    }

    // Move the piece
    remove_piece(from);
    add_piece(side_to_move, p, to);

    // Switch turns
    side_to_move = (side_to_move == WHITE) ? BLACK : WHITE;
}


void ChessBoard::undo_move(const Move& move, bool interactive) {
    // Get the moved piece and its color
    Color mover = (side_to_move == WHITE) ? BLACK : WHITE; // Undoing move, so switch turn back
    Piece p = move.piece;

    // Undo promotion: If the piece was promoted, revert it back to a pawn
    if (move.type == PROMOTION) {
        p = PAWN;
    }

    // Restore captured piece (if there was one)
    if (move.captured_piece != NO_PIECE) {
        Color captured_color = (mover == WHITE) ? BLACK : WHITE;
        add_piece(captured_color, move.captured_piece, move.to);
    }

    // Undo castling
    if (move.type == CASTLING) {
        if (mover == WHITE) {
            if (move.to == G1) { // Kingside
                remove_piece(F1);
                add_piece(WHITE, ROOK, H1);
            } else if (move.to == C1) { // Queenside
                remove_piece(D1);
                add_piece(WHITE, ROOK, A1);
            }
        } else if (mover == BLACK) {
            if (move.to == G8) { // Kingside
                remove_piece(F8);
                add_piece(BLACK, ROOK, H8);
            } else if (move.to == C8) { // Queenside
                remove_piece(D8);
                add_piece(BLACK, ROOK, A8);
            }
        }
    }

    // Undo en passant
    if (move.type == EN_PASSANT) {
        if (mover == WHITE) {
            add_piece(BLACK, PAWN, static_cast<Square>(move.to - 8)); // Restore Black pawn
        } else {
            add_piece(WHITE, PAWN, static_cast<Square>(move.to + 8)); // Restore White pawn
        }
    }

    // Move the piece back to its original square
    remove_piece(move.to);
    add_piece(mover, p, move.from);

    // Switch turns back
    side_to_move = mover;
}
