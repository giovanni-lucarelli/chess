#pragma once
#include <cstdint>
#include <string>
#include <iostream>
#include "chessboard.hpp"

// 32-bit unsigned integer (represents a 5x5 chessboard)
using U32 = uint32_t;

// Colors and pieces
enum Color { WHITE, BLACK, NO_COLOR };
enum Piece { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE };

// Chessboard squares (A1 = 0, E5 = 24)
// this can be usefull for the bitboard operations; each one is an integer
enum Square {
  A1, B1, C1, D1, E1,
  A2, B2, C2, D2, E2,
  A3, B3, C3, D3, E3,
  A4, B4, C4, D4, E4,
  A5, B5, C5, D5, E5,
  NO_SQUARE
};

inline Square& operator++(Square& sq) {
    sq = static_cast<Square>(static_cast<int>(sq) + 1);
    return sq;
}

// Utility function to convert a position to a number

// convert number to position
inline std::string square_to_string(Square sq) {
  std::string files = "abcde";
  std::string ranks = "12345";
  return std::string(1, files[sq % 5]) + std::string(1, ranks[sq / 5]);
}

inline std::string color_to_string(Color c) {
  return c == WHITE ? "WHITE" : "BLACK";
}

inline std::string piece_to_string(Piece p) {
  switch (p) {
    case PAWN: return "PAWN";
    case KNIGHT: return "KNIGHT";
    case BISHOP: return "BISHOP";
    case ROOK: return "ROOK";
    case QUEEN: return "QUEEN";
    case KING: return "KING";
    default: return "INVALID";
  }
}