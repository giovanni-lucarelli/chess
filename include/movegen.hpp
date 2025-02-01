#pragma once
#include "chessboard.hpp"
#include "move.hpp"
#include <vector>

class MoveGenerator {
public:
    static std::vector<Move> generate_moves(const ChessBoard& board);

private:
    static void generate_pawn_moves(const ChessBoard& board, std::vector<Move>& moves);
    static void generate_knight_moves(const ChessBoard& board, std::vector<Move>& moves);
    static void generate_bishop_moves(const ChessBoard& board, std::vector<Move>& moves);
    static void generate_rook_moves(const ChessBoard& board, std::vector<Move>& moves);
    static void generate_queen_moves(const ChessBoard& board, std::vector<Move>& moves);
    static void generate_king_moves(const ChessBoard& board, std::vector<Move>& moves);
};
