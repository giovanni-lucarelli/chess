#include "game.hpp"
#include "types.hpp"
#include "move.hpp"
#include "chessboard.hpp"
#include <iostream>
#include <chrono>
#include <sstream>
#include <cctype>
#include <algorithm>

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

void Game::update_check() {
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

    // Handle castling separately to avoid multiple entries
    if (p == KING && (from == E1 || from == E8)) {  
        if (mover == WHITE) {
            if (castling_rights[WHITE][1] && !board.is_occupied(F1) && !board.is_occupied(G1) &&
                board.is_path_clear(E1, G1)) {
                moves.push_back({mover, p, from, G1, CASTLING});
            }
            if (castling_rights[WHITE][0] && !board.is_occupied(B1) && !board.is_occupied(C1) &&
                !board.is_occupied(D1) && board.is_path_clear(E1, C1)) {
                moves.push_back({mover, p, from, C1, CASTLING});
            }
        } else if (mover == BLACK) {
            if (castling_rights[BLACK][1] && !board.is_occupied(F8) && !board.is_occupied(G8) &&
                board.is_path_clear(E8, G8)) {
                moves.push_back({mover, p, from, G8, CASTLING});
            }
            if (castling_rights[BLACK][0] && !board.is_occupied(B8) && !board.is_occupied(C8) &&
                !board.is_occupied(D8) && board.is_path_clear(E8, C8)) {
                moves.push_back({mover, p, from, C8, CASTLING});
            }
        }
    }

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

                break; 
            default:
                break;
        }
        if (valid) {
            MoveType moveType = (target_info.first != NO_COLOR) ? CAPTURE : NORMAL;
            moves.push_back({mover, p, from, static_cast<Square>(to), moveType, target_info.second});
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

    // Second: simulate the move and check king safety.
    Game game_copy = *this;
    game_copy.do_move(input_move);
    game_copy.update_check();
    bool next_white_check = game_copy.get_check(WHITE);
    bool next_black_check = game_copy.get_check(BLACK);

    // The move is legal if it does not leave the mover's king in check.
    if (side_to_move == WHITE)
        return !next_white_check;
    else
        return !next_black_check;
    
    return true;
}

std::vector<Move> Game::legal_moves(Square from) const {
    std::vector<Move> moves{};
    auto piece_info = board.get_piece_on_square(from);
    if (piece_info.first == NO_COLOR)
        return moves;

    for (auto move : pseudo_legal_moves(from)) {
        if (is_move_legal(move))
            // std::cout << "Debug: Legal move: " << square_to_string(move.from) << " -> " << square_to_string(move.to) << std::endl;
            moves.push_back(move);
    }
    return moves;
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

bool Game::is_insufficient_material() const {

    int wb = board.count(BISHOP, WHITE);
    int wn = board.count(KNIGHT, WHITE);
    int bb = board.count(BISHOP, BLACK);
    int bn = board.count(KNIGHT, BLACK);

    // K vs K
    if ((wb + wn + bb + bn) == 0) return true;

    // K vs (B or N)
    if ((wb + wn) == 1 && (bb + bn) == 0) return true;
    if ((bb + bn) == 1 && (wb + wn) == 0) return true;

    // two horses 
    if (wn >= 2 && wb == 0 && bb == 0 && bn == 0) return true;
    if (bn >= 2 && bb == 0 && wb == 0 && wn == 0) return true;

    // otherwise don't declare insufficient
    return false;
}


void Game::do_move(Move& move) {
    Square from = move.from;
    Square to = move.to;
    Piece p = move.piece;
    MoveType type = move.type;

    // Ensure the move is legal
    if (p == NO_PIECE) {
        std::cerr << "Error: No piece on the selected square!\n";
        return;
    }

    // Store captured piece before removing it
    move.captured_piece = NO_PIECE;  // Default to no piece captured
    if (board.is_occupied(to) && type != CASTLING) {
        move.captured_piece = board.get_piece_on_square(to).second;  // Store captured piece
        board.remove_piece(to);
    }

    // Handle en passant
    if (type == EN_PASSANT) {
        Square captured_pawn_square = static_cast<Square>(to + (side_to_move == WHITE ? -8 : 8));
        move.captured_piece = PAWN;  // Store captured pawn
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
            move.promoted_to = p;  // Store promoted piece
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
    update_check();  // Update check status after the move
}


void Game::undo_move(Move& move) {
    // Determine who made the move
    Color mover = (side_to_move == WHITE) ? BLACK : WHITE; // Previous turn
    Piece p = move.piece;

    if (move.type == EN_PASSANT) {
        // Remove the pawn from move.to (D6)
        board.remove_piece(move.to);

        // Restore moved pawn to move.from (E5)
        board.add_piece(mover, p, move.from);

        // Restore captured pawn on the correct square (D5)
        Color captured_color = (mover == WHITE) ? BLACK : WHITE;
        Square captured_pawn_square = (mover == WHITE) ? static_cast<Square>(move.to - 8) : static_cast<Square>(move.to + 8);
        board.add_piece(captured_color, PAWN, captured_pawn_square);

    } else {
        // Normal moves (including capture, promotion, castling) undo logic
        board.remove_piece(move.to);
        board.add_piece(mover, p, move.from);

        // Restore captured piece (if any)
        if (move.captured_piece != NO_PIECE) {
            Color captured_color = (mover == WHITE) ? BLACK : WHITE;
            board.add_piece(captured_color, move.captured_piece, move.to);
        }

        // Undo promotion (restore pawn)
        if (move.type == PROMOTION) {
            board.remove_piece(move.from);  // Remove promoted piece
            board.add_piece(mover, PAWN, move.from);
        }

        // Undo castling (restore rook)
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
        }
    }

    // Switch turns back after move is undone
    side_to_move = mover;
}

Move Game::parse_move(const std::string& from, const std::string& to) const {
    Square from_sq = static_cast<Square>(8 * (from[1] - '1') + (from[0] - 'a'));
    Square to_sq = static_cast<Square>(8 * (to[1] - '1') + (to[0] - 'a'));

    // Get the moving piece
    Color color = board.get_piece_on_square(from_sq).first;
    Piece piece = board.get_piece_on_square(from_sq).second;
    
    Move move(color, piece, from_sq, to_sq, *this);

    if (!is_move_legal(move))
        throw std::runtime_error("parse_move: illegal move");
    
    return move;
}

Move Game::parse_action_to_move(int action) const {
    // Action encoding: action = from_square * 64 + to_square
    // With action range 0-4095 (64*64)
    if (action < 0 || action >= 4096) {
        throw std::runtime_error("parse_action_to_move: action out of range [0, 4095]");
    }
    
    Square from_sq = static_cast<Square>(action / 64);
    Square to_sq = static_cast<Square>(action % 64);
    
    // Check if from square has a piece
    if (!board.is_occupied(from_sq)) {
        throw std::runtime_error("parse_action_to_move: no piece on from square");
    }
    
    // Get the moving piece
    Color color = board.get_piece_on_square(from_sq).first;
    Piece piece = board.get_piece_on_square(from_sq).second;
    
    // Create the move
    Move move(color, piece, from_sq, to_sq, *this);
    
    // Check if the move is legal
    if (!is_move_legal(move)) {
        throw std::runtime_error("parse_action_to_move: illegal move");
    }
    
    return move;
}

void Game::play() {
    
    while (true) {

        turn += 1;
        board.print();

        // Check if the king is in check
        update_check();
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
        board.print();

        char undo_bool;
        std::cout << "Undo move? (y/n): ";
        std::cin >> undo_bool;
        if(undo_bool == 'y') {
            undo_move(move);
            std::cout << "Move undone\n";
            board.print();

        }
        
    }

}


namespace {
    inline Square sq_of(int file, int rank) {  // file,rank are 0..7; rank 0=a1.. rank 7=a8
        return static_cast<Square>(rank * 8 + file);
    }
    inline int file_of(Square s) { return static_cast<int>(s) % 8; }
    inline int rank_of(Square s) { return static_cast<int>(s) / 8; }

    inline bool fen_to_square(const std::string& s, Square& out) {
        if (s.size() != 2) return false;
        char f = s[0], r = s[1];
        if (f < 'a' || f > 'h' || r < '1' || r > '8') return false;
        int file = f - 'a';
        int rank = r - '1';
        out = sq_of(file, rank);
        return true;
    }
    inline std::string square_to_fen(Square sq) {
        char f = 'a' + file_of(sq);
        char r = '1' + rank_of(sq);
        return std::string{f, r};
    }

    inline bool char_to_piece(char c, Color& col, Piece& pt) {
        col = std::isupper(static_cast<unsigned char>(c)) ? WHITE : BLACK;
        char u = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        switch (u) {
            case 'P': pt = PAWN;   return true;
            case 'N': pt = KNIGHT; return true;
            case 'B': pt = BISHOP; return true;
            case 'R': pt = ROOK;   return true;
            case 'Q': pt = QUEEN;  return true;
            case 'K': pt = KING;   return true;
            default: return false;
        }
    }
    inline char piece_to_char(Color col, Piece pt) {
        char c = '?';
        switch (pt) {
            case PAWN:   c = 'p'; break;
            case KNIGHT: c = 'n'; break;
            case BISHOP: c = 'b'; break;
            case ROOK:   c = 'r'; break;
            case QUEEN:  c = 'q'; break;
            case KING:   c = 'k'; break;
            default:     c = '?'; break;
        }
        return (col == WHITE) ? static_cast<char>(std::toupper(c)) : c;
    }
}

void Game::reset_from_fen(const std::string& fen) {
    // Clear state
    board.clear();
    white_check = black_check = false;
    checkmate = false;

    // Split FEN into fields
    std::istringstream iss(fen);
    std::string placement, stm, castling, ep, halfmove, fullmove;
    iss >> placement >> stm >> castling >> ep >> halfmove >> fullmove;
    if (placement.empty() || stm.empty()) {
        throw std::runtime_error("Invalid FEN: missing fields");
    }

    // 1) Piece placement (ranks 8 -> 1)
    int rank = 7, file = 0;
    for (char c : placement) {
        if (c == '/') {
            if (file != 8) throw std::runtime_error("Invalid FEN: rank too short");
            rank--; file = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            file += (c - '0');
            if (file > 8) throw std::runtime_error("Invalid FEN: too many files in rank");
        } else {
            Color col; Piece pt;
            if (!char_to_piece(c, col, pt))
                throw std::runtime_error("Invalid FEN: bad piece char");
            if (file >= 8 || rank < 0) throw std::runtime_error("Invalid FEN: coord overflow");
            board.add_piece(col, pt, sq_of(file, rank));  // <-- adapt if your API differs
            file++;
        }
    }
    if (rank != 0 || file != 8) {
        // After parsing, we should be at end of rank 1
        // (rank==0 and file==8, because the loop stops after finishing last rank)
        // Many engines allow rank==0,file==8 or rank==-1,file==0; adjust if you prefer.
    }

    // 2) Side to move
    side_to_move = (stm == "w" || stm == "W") ? WHITE : BLACK;

    // 3) Castling rights (your layout is [Color][Queenside, Kingside])
    castling_rights[WHITE][0] = castling_rights[WHITE][1] = false;
    castling_rights[BLACK][0] = castling_rights[BLACK][1] = false;
    if (!castling.empty() && castling != "-") {
        for (char c : castling) {
            if (c == 'K') castling_rights[WHITE][1] = true;   // White kingside
            if (c == 'Q') castling_rights[WHITE][0] = true;   // White queenside
            if (c == 'k') castling_rights[BLACK][1] = true;   // Black kingside
            if (c == 'q') castling_rights[BLACK][0] = true;   // Black queenside
        }
    }

    // 4) En passant square
    if (!ep.empty() && ep != "-") {
        Square epsq;
        if (!fen_to_square(ep, epsq)) throw std::runtime_error("Invalid FEN: bad en-passant square");
        en_passant_square = epsq;
    } else {
        // If you have a NO_SQUARE enum, use it. Otherwise pick a sentinel you never treat as real.
        en_passant_square = NO_SQUARE; // sentinel "no square" as you noted
    }

    // Optionally: parse halfmove/fullmove if you later add those fields.
    // We ignore them here.

    // Update check flags based on the new position
    update_check();
}

std::string Game::to_fen() const {
    // 1) Piece placement (ranks 8 -> 1)
    std::string placement;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            Square sq = sq_of(file, rank);
            Color c; Piece p;
            if (board.is_occupied(sq)) {
                std::tie(c, p) = board.get_piece_on_square(sq);
                // piece_at(...) should fill (c,p) and return true; adapt if your API differs
                // if (!board.add_piece(c, p,sq)) {
                //     throw std::runtime_error("Board says occupied but piece_at failed");
                // }
                if (empty > 0) { placement += char('0' + empty); empty = 0; }
                placement += piece_to_char(c, p);
            } else {
                empty++;
            }
        }
        if (empty > 0) placement += char('0' + empty);
        if (rank > 0) placement += '/';
    }

    // 2) Side to move
    std::string stm = (side_to_move == WHITE) ? "w" : "b";

    // 3) Castling rights
    std::string cast;
    if (castling_rights[WHITE][1]) cast += 'K';
    if (castling_rights[WHITE][0]) cast += 'Q';
    if (castling_rights[BLACK][1]) cast += 'k';
    if (castling_rights[BLACK][0]) cast += 'q';
    if (cast.empty()) cast = "-";

    // 4) En passant
    std::string ep = "-";
    // If you have a NO_SQUARE constant, check that instead of H8.
    if (en_passant_square != NO_SQUARE) {
        ep = square_to_fen(en_passant_square);
    }

    // 5) Halfmove/fullmove – if you don’t track them, emit "0 1"
    return placement + " " + stm + " " + cast + " " + ep + " 0 1";
}
