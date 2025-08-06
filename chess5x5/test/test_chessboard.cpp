#include <gtest/gtest.h>
#include "chessboard.hpp"

class ChessBoardTest : public ::testing::Test {
protected:
    ChessBoard board;

    void SetUp() override {
        board.reset();
    }
};

// Test board initialization for 5x5 endgame
TEST_F(ChessBoardTest, ResetBoard) {
    // Black King on A5 (square 20) = 1U << 20
    EXPECT_EQ(board.get_pieces(BLACK, KING), 1U << 20);
    // White Queen on C3 (square 12) = 1U << 12
    EXPECT_EQ(board.get_pieces(WHITE, QUEEN), 1U << 12);
    // White Rook on E2 (square 9) = 1U << 9
    EXPECT_EQ(board.get_pieces(WHITE, ROOK), 1U << 9);
    // White King on E1 (square 4) = 1U << 4
    EXPECT_EQ(board.get_pieces(WHITE, KING), 1U << 4);
}

// Test piece occupancy for 5x5 board
TEST_F(ChessBoardTest, IsOccupied) {
    EXPECT_TRUE(board.is_occupied(E1)); // White King (square 4)
    EXPECT_TRUE(board.is_occupied(C3)); // White Queen (square 12)
    EXPECT_TRUE(board.is_occupied(E2)); // White Rook (square 9)
    EXPECT_TRUE(board.is_occupied(A5)); // Black King (square 20)
    EXPECT_FALSE(board.is_occupied(A1)); // Empty square (square 0)
}