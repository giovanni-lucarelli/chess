// Description: Defines the Move struct, which represents a move in the game.

#pragma once
#include "types.hpp"

// Flags for special moves
enum MoveType {
    NORMAL,
    CASTLING,
    EN_PASSANT,
    PROMOTION,
    DOUBLE_PAWN_PUSH
};

struct Move {
    Square from;       // Starting square
    Square to;         // Target square
    Piece promoted_to; // For pawn promotions (e.g., QUEEN)
    MoveType type;     // Flags for special moves (en passant, castling, etc.)

    // Store captured piece (for undoing moves)?
    Piece captured_piece;

    // Default constructor
    Move() : from(Square::NO_SQUARE), to(Square::NO_SQUARE), 
             promoted_to(Piece::NO_PIECE), type(MoveType::NORMAL), 
             captured_piece(Piece::NO_PIECE) {}

    // Set move with starting and target squares
    Move(Square from, Square to) : from(from), to(to), 
                                   promoted_to(Piece::NO_PIECE), type(MoveType::NORMAL), 
                                   captured_piece(Piece::NO_PIECE) {}

    // Overload == operator for move comparison
    bool operator==(const Move& other) const {
        return from == other.from && to == other.to && 
               promoted_to == other.promoted_to && type == other.type;
    }
};


