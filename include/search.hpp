#ifndef SEARCH_HPP
#define SEARCH_HPP

#include "chessboard.hpp"
#include "move.hpp"
#include "movegen.hpp"

// Evaluation and search functions
int evaluate(const ChessBoard& board);
int alpha_beta(ChessBoard& board, int depth, int alpha, int beta, bool maximizingPlayer);
Move find_best_move(ChessBoard& board, int depth);

#endif
