#pragma once
#include "types.hpp"
#include <iostream>
#include <string>

class Bitboard {

public:
    // Set/Clear bits
    static void set_bit(U32& board, Square sq) {
        board |= (1U << static_cast<int>(sq));
    }

    static void clear_bit(U32& board, Square sq) {
        board &= ~(1U << static_cast<int>(sq));
    }

    static bool get_bit(U32 board, Square sq) {
        return (board & (1U << static_cast<int>(sq))) != 0;
    }

    static int count_bits(U32 board) {
        int count = 0;
        while (board) {
            count++;
            board &= board - 1; // Clear least significant bit
        }
        return count;
    }

    // Print board with rank/file labels
    static void print(U32 board) {
        for (int rank = 4; rank >= 0; rank--) {
            std::cout << (rank + 1) << " ";
            for (int file = 0; file < 5; file++) {
                int sq = rank * 5 + file;
                std::cout << (get_bit(board, static_cast<Square>(sq)) ? "X " : "  ");
            }
            std::cout << "\n";
        }
        std::cout << "  a b c d e\n\n";
    }

};
