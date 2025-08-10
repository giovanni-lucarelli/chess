#pragma once
#include <iostream>
#include "chessboard.hpp"
#include "game.hpp"

class Game;  // Forward declaration

// Flags for special moves
enum MoveType {
	NORMAL,
	CASTLING,
	EN_PASSANT,
	PROMOTION,
	DOUBLE_PAWN_PUSH,
	CAPTURE
};
  
inline std::string movetype_to_string(MoveType type) {
	switch (type) {
		case NORMAL: return "NORMAL";
		case CASTLING: return "CASTLING";
		case EN_PASSANT: return "EN_PASSANT";
		case PROMOTION: return "PROMOTION";
		case DOUBLE_PAWN_PUSH: return "DOUBLE_PAWN_PUSH";
		case CAPTURE: return "CAPTURE";
		default: return "INVALID";
	}
}
  
class Move {

public:

    Color color;       // Color of the piece moved
    Piece piece;       // Piece moved
    Square from;       // Starting square
    Square to;         // Target square
    MoveType type;     // Flags for special moves (en passant, castling, etc.)
    // Store captured piece (for undoing moves)?
    Piece captured_piece;
    Piece promoted_to; // For pawn promotions (e.g., QUEEN)
  
    // Default constructor
    Move() : color(NO_COLOR), piece(NO_PIECE), from(NO_SQUARE), to(NO_SQUARE), 
    type(MoveType::NORMAL), captured_piece(NO_PIECE), promoted_to(NO_PIECE) {};
    
    // Parametrized with custom MoveType

    Move(Color color, Piece piece, Square from, Square to, MoveType type, Piece captured_piece = NO_PIECE, Piece promoted_to = NO_PIECE) 
        : color(color), piece(piece), from(from), to(to), type(type), 
          captured_piece(captured_piece), promoted_to(promoted_to) {}

    // Parametrized with automatic MoveType
    Move(Color color, Piece piece, Square from, Square to, const Game& game);

    void print();

    bool operator==(Move const& o) const noexcept {
        return from      == o.from
            &&  to        == o.to
            &&  type      == o.type
            &&  promoted_to == o.promoted_to;
    }

    std::string to_string() const {
    // This assumes your Square enum can be converted to a string,
    // which is likely the case for your board printing to work.
    return square_to_string(from) + square_to_string(to);
    }
  
};