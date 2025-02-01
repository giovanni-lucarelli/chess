// bitboard.hpp
#pragma once
#include "types.hpp"
#include <iostream>

class Bitboard {

public:
    // Set/Clear bits
    static void set_bit(U64& board, Square sq) {
        board |= (1ULL << static_cast<int>(sq));
    }

    static void clear_bit(U64& board, Square sq) {
        board &= ~(1ULL << static_cast<int>(sq));
    }

    static bool get_bit(U64 board, Square sq) {
        return (board & (1ULL << static_cast<int>(sq))) != 0;
    }

    static int count_bits(U64 board) {
        int count = 0;
        while (board) {
            count++;
            board &= board - 1; // Clear least significant bit
        }
        return count;
    }

    // Print board with rank/file labels
    static void print(U64 board) {
        for (int rank = 7; rank >= 0; rank--) {
            std::cout << (rank + 1) << " ";
            for (int file = 0; file < 8; file++) {
                int sq = rank * 8 + file;
                std::cout << (get_bit(board, static_cast<Square>(sq)) ? "X " : ". ");
            }
            std::cout << "\n";
        }
        std::cout << "  a b c d e f g h\n\n";
    }
};
