#include "search.hpp"
#include "movegen.hpp"
#include <vector>
#include <algorithm>
#include <map>

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
int alpha_beta(ChessBoard& board1, int depth, int alpha, int beta, bool maximizingPlayer) {
    
    ChessBoard board = board1;
    if (depth == 0 || board.is_game_over()) {
        return evaluate(board);
    }

    MoveGenerator moveGen;
    std::vector<Move> legal_moves = moveGen.generate_legal_moves(board);

    if (maximizingPlayer) {
        int maxEval = -100000;
        for (Move move : legal_moves) {
            board.move_piece(move.from, move.to);
            int eval = alpha_beta(board, depth - 1, alpha, beta, false);
            board.undo_move(move.from, move.to);
            maxEval = std::max(maxEval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) break;  // Pruning
        }
        return maxEval;
    } else {
        int minEval = 100000;
        for (Move move : legal_moves) {
            board.move_piece(move.from, move.to);
            int eval = alpha_beta(board, depth - 1, alpha, beta, true);
            board.undo_move(move.from, move.to);
            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) break;  // Pruning
        }
        return minEval;
    }
}

// --- Find the Best Move ---
Move find_best_move(ChessBoard& board1, int depth) {
    ChessBoard board = board1;
    Move bestMove;
    int bestEval = -100000;
    int alpha = -100000;
    int beta = 100000;

    MoveGenerator moveGen;
    std::vector<Move> legal_moves = moveGen.generate_legal_moves(board);

    for (Move move : legal_moves) {
        board.move_piece(move.from, move.to);
        int eval = alpha_beta(board, depth - 1, alpha, beta, false);
        board.undo_move(move.from, move.to);
        if (eval > bestEval) {
            bestEval = eval;
            bestMove = move;
        }
    }
    
    return bestMove;
}
