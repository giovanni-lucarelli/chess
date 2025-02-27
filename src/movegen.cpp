#include "movegen.hpp"
#include "types.hpp"
#include <iostream>
#include <map>

// Generates all legal moves for the current position
std::vector<Move> MoveGenerator::generate_legal_moves(const ChessBoard& board){
    std::vector<Move> moves;
    for (int i = 0; i < 64; i++){
        Square from = static_cast<Square>(i);
        if (board.is_occupied(from) && board.get_piece_on_square(from).first == board.get_side_to_move()){
            std::vector<Move> targets = board.legal_moves(from);
            for (Move target : targets){
                Move move(target.from, target.to, target.piece);
                move.color = board.get_side_to_move();
                if (board.is_move_legal(move)){
                    moves.push_back(move);
                }
            }
        }
    }
    return moves;
}

// print all moves: for debugging purposes
void MoveGenerator::print_moves(const std::vector<Move>& moves){
    for (Move move : moves){
        std::cout << "Move " << piece_to_string(move.piece) << " from " << square_to_string(move.from) << " to " << square_to_string(move.to) << std::endl;
    }
}
