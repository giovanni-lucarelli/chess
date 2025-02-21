// Description: Generate pseudo-legal and legal moves for a given board state.

#pragma once
#include "chessboard.hpp"
#include "move.hpp"
#include <vector>

class MoveGenerator {
public:
    // Generate legal moves (filter out moves that leave king in check)
    std::vector<Move> generate_legal_moves(const ChessBoard& board);

    // helper: print all moves
    static void print_moves(const std::vector<Move>& moves);

};