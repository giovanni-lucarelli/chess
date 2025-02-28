#include "search.hpp"
#include "types.hpp"
#include "chessboard.hpp"
#include <vector>
#include <algorithm>
#include <map>

// Generates all legal moves for the current position
// ??? Maybe not necessay, since there is legal_moves() in ChessBoard class with both from and colors
// std::vector<Move> generate_legal_moves(const ChessBoard& board) {
//     std::vector<Move> moves;
//     for (int i = 0; i < 64; i++) {
//         Square from = static_cast<Square>(i);
//         if (board.is_occupied(from) && board.get_piece_on_square(from).first == board.get_side_to_move()) {
//             std::vector<Move> targets = board.legal_moves(from);
//             for (const Move& target : targets) { // Use const reference to avoid copies
//                 if (board.is_move_legal(target)) {
//                     moves.push_back(target); // Push the original target move
//                 }
//             }
//         }
//     }
//     return moves;
// }

// print all moves: for debugging purposes
void print_moves(const std::vector<Move>& moves) {
    for (const Move& move : moves) {
        std::cout << "Move " << piece_to_string(move.piece) << " from " 
                  << square_to_string(move.from) << " to " 
                  << square_to_string(move.to);
        if (move.promoted_to != NO_PIECE) {
            std::cout << " promote to " << piece_to_string(move.promoted_to);
        }
        std::cout << std::endl;
    }
}

// Not working correctly
int evaluate(const ChessBoard& board) {
    int score = 0;
    const std::map<Piece, int> piece_values = {
        {PAWN, 100}, {KNIGHT, 300}, {BISHOP, 300},
        {ROOK, 500}, {QUEEN, 900}, {KING, 10000}
    };

    for (int i = 0; i < 64; i++) {
        Square square = static_cast<Square>(i);
        if (board.is_occupied(square)) {
            auto piece = board.get_piece_on_square(square);
            int value = piece_values.at(piece.second);
            score += (piece.first == WHITE) ? value : -value;
        }
    }
    return score;
}

// --- Alpha-Beta Pruning Search ---
int alpha_beta(ChessBoard& board, int depth, int alpha, int beta, bool maximizingPlayer) {
    // if (depth == 0 || board.is_game_over()) {
    if (depth == 0) {
        return evaluate(board);
    }

    std::vector<Move> legal_moves = board.legal_moves(board.get_side_to_move());

    if (maximizingPlayer) {
        int maxEval = -100000;
        for (Move move : legal_moves) {
            Piece captured_piece = board.get_piece_on_square(move.to).second;  // Track captured piece
            Piece promoted_piece = (move.promoted_to ? move.promoted_to : NO_PIECE);  // Track promotion

            board.do_move(move);
            int eval = alpha_beta(board, depth - 1, alpha, beta, false);
            board.undo_move(move);

            maxEval = std::max(maxEval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) break;  // Pruning
        }
        return maxEval;
    } else {
        int minEval = 100000;
        for (Move move : legal_moves) {
            Piece captured_piece = board.get_piece_on_square(move.to).second;
            Piece promoted_piece = (move.promoted_to ? move.promoted_to : NO_PIECE);

            board.do_move(move);
            int eval = alpha_beta(board, depth - 1, alpha, beta, true);
            board.undo_move(move);

            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) break;  // Pruning
        }
        return minEval;
    }
}



// --- Find the Best Move ---
Move find_best_move(ChessBoard& board, int depth) {
    Move bestMove;
    int bestEval = -100000;
    int alpha = -100000;
    int beta = 100000;

    std::vector<Move> legal_moves = board.legal_moves(board.get_side_to_move());

    for (Move move : legal_moves) {
        Piece captured_piece = board.get_piece_on_square(move.to).second;
        Piece promoted_piece = (move.promoted_to ? move.promoted_to : NO_PIECE);

        board.do_move(move);
        int eval = alpha_beta(board, depth - 1, alpha, beta, false);
        board.undo_move(move);

        if (eval > bestEval) {
            bestEval = eval;
            bestMove = move;
        }
    }
    
    return bestMove;
}


