#pragma once
#include "chessboard.hpp"
#include <vector>



// Generate legal moves (filter out moves that leave king in check)
std::vector<Move> generate_legal_moves(const ChessBoard& board);

// helper: print all moves
static void print_moves(const std::vector<Move>& moves);

// Evaluation and search functions
int evaluate(const ChessBoard& board);
int alpha_beta(ChessBoard& board, int depth, int alpha, int beta, bool maximizingPlayer);
Move find_best_move(ChessBoard& board, int depth);
