// Description: Generate pseudo-legal and legal moves for a given board state.

#pragma once
#include "chessboard.hpp"
#include "move.hpp"
#include <vector>

class MoveGenerator {
public:
    // Generate all pseudo-legal moves (may leave king in check)
    static std::vector<Move> generate_pseudo_legal_moves(const ChessBoard& board);

    // Generate legal moves (filter out moves that leave king in check)
    static std::vector<Move> generate_legal_moves(ChessBoard& board);

private:
    // Piece-specific generation helpers
    static void generate_pawn_moves(std::vector<Move>& moves, const ChessBoard& board);
    static void generate_knight_moves(std::vector<Move>& moves, const ChessBoard& board);
    static void generate_king_moves(std::vector<Move>& moves, const ChessBoard& board);
    static void generate_slider_moves(std::vector<Move>& moves, const ChessBoard& board);

    // Helper for sliding pieces (bishops/rooks/queens) see Magic Bitboards
    static U64 get_slider_attacks(Square sq, Piece piece, U64 all_pieces);
};
