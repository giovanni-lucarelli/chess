#pragma once
#include <cstdint>
#include <string>

// 64-bit unsigned integer (represents a chessboard)
using U64 = uint64_t;

// Colors and pieces
enum Color { WHITE, BLACK, NO_COLOR };
enum Piece { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE };

// Chessboard squares (A1 = 0, H8 = 63)
// this can be usefull for thebitboard operations; each one is an integer
enum Square {
  A1, B1, C1, D1, E1, F1, G1, H1,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A8, B8, C8, D8, E8, F8, G8, H8,
  NO_SQUARE // Add this line to define NO_SQUARE
};

inline Square& operator++(Square& sq) {
    sq = static_cast<Square>(static_cast<int>(sq) + 1);
    return sq;
}

// convert number to position
inline std::string square_to_string(Square sq) {
  std::string files = "ABCDEFGH";
  std::string ranks = "12345678";
  return std::string(1, files[sq % 8]) + std::string(1, ranks[sq / 8]);
}


