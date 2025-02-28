#include "search.hpp"
#include "types.hpp"
#include "game.hpp"
#include "move.hpp"
#include <vector>
#include <algorithm>
#include <map>

std::vector<Move> generate_legal_moves(const Game& game) {
    std::vector<Move> moves;
    auto board = game.get_board();
    for (int i = 0; i < 64; i++) {
        Square from = static_cast<Square>(i);
        if (board.is_occupied(from) && board.get_piece_on_square(from).first == game.get_side_to_move()) {
            std::vector<Move> targets = game.legal_moves(from);
            moves.insert(moves.end(), targets.begin(), targets.end()); // No need for extra check
        }
    }
    return moves;
}

void print_moves(const std::vector<Move>& moves) {
    for (const Move& move : moves) {
        std::cout << "Move " << color_to_string(move.color) << piece_to_string(move.piece) << " from " 
                  << square_to_string(move.from) << " to " 
                  << square_to_string(move.to);
        if (move.promoted_to != NO_PIECE) {
            std::cout << " promote to " << piece_to_string(move.promoted_to);
        }
        std::cout << std::endl;
    }
}

int evaluate(const Game& game) {
    static const int piece_values[6] = {100, 300, 300, 500, 900, 10000};

    static const int pawn_table[64] = {
         0,  5,  5,  0,  5, 10, 50,  0,
         0, 10, -5,  0,  5, 15, 50,  0,
         0, 10,-10, 20, 30, 30, 50,  0,
         5, 10,  5, 25, 35, 40, 50,  5,
         5, 10, 10, 25, 35, 40, 50,  5,
         5, 10, 20, 30, 30, 30, 50,  5,
         0,  0,  0,  0,  0,  0, 50,  0,
         0,  0,  0,  0,  0,  0, 50,  0
    };

    auto board = game.get_board();
    int score = 0;

    for (int i = 0; i < 64; i++) {
        Square square = static_cast<Square>(i);
        if (!board.is_occupied(square)) continue;

        auto piece = board.get_piece_on_square(square);
        int piece_value = piece_values[piece.second];
        int square_bonus = (piece.second == PAWN) ? pawn_table[i] : 0;

        int final_value = piece_value + square_bonus;
        score += (piece.first == WHITE) ? final_value : -final_value;
    }

    return score;
}

void sort_moves(std::vector<Move>& moves) {
    static const std::map<Piece, int> piece_values = {
        {PAWN, 100}, {KNIGHT, 300}, {BISHOP, 300},
        {ROOK, 500}, {QUEEN, 900}, {KING, 10000},
        {NO_PIECE, 0} // for captures
    };
    

    std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
        if (a.type == CAPTURE && b.type != CAPTURE) return true;
        if (a.type != CAPTURE && b.type == CAPTURE) return false;

        int victimA = piece_values.at(a.captured_piece);
        int attackerA = piece_values.at(a.piece);
        int victimB = piece_values.at(b.captured_piece);
        int attackerB = piece_values.at(b.piece);

        return (victimA > victimB) || (victimA == victimB && attackerA < attackerB);
    });
}


int quiescence(Game& game, int alpha, int beta) {
    int stand_pat = evaluate(game);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    std::vector<Move> captures = generate_legal_moves(game);

    captures.erase(std::remove_if(captures.begin(), captures.end(), [](const Move& move) {
        return move.type != CAPTURE;
    }), captures.end());

    sort_moves(captures);

    for (Move& move : captures) {
        game.do_move(move);
        int score = -quiescence(game, -beta, -alpha);
        game.undo_move(move);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}


int alpha_beta(Game& game, int depth, int alpha, int beta, bool maximizingPlayer) {
    if (depth == 0 || game.is_game_over()) {
        return quiescence(game, alpha, beta);
    }

    std::vector<Move> legal_moves = generate_legal_moves(game);
    sort_moves(legal_moves);

    if (maximizingPlayer) {
        int maxEval = -100000;
        for (Move move : legal_moves) {
            game.do_move(move);
            int eval = alpha_beta(game, depth - 1, alpha, beta, false);
            game.undo_move(move);

            maxEval = std::max(maxEval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) break;  
        }
        return maxEval;
    } else {
        int minEval = 100000;
        for (Move move : legal_moves) {
            game.do_move(move);
            int eval = alpha_beta(game, depth - 1, alpha, beta, true);
            game.undo_move(move);

            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) break;  
        }
        return minEval;
    }
}


Move find_best_move(Game& game, int max_depth) {
    Move bestMove;
    int bestEval = -100000;
    int alpha = -100000;
    int beta = 100000;

    for (int depth = 1; depth <= max_depth; depth++) {  
        std::vector<Move> legal_moves = generate_legal_moves(game);
        sort_moves(legal_moves);

        for (Move move : legal_moves) {
            game.do_move(move);
            int eval = alpha_beta(game, depth - 1, alpha, beta, false);
            game.undo_move(move);

            if (eval > bestEval) {
                bestEval = eval;
                bestMove = move;
                alpha = std::max(alpha, eval);
            }
        }
    }

    return bestMove;
}
