#include <gtest/gtest.h>
#include "chessboard.hpp"

class ChessBoardTest : public ::testing::Test {
protected:
    ChessBoard board;

    void SetUp() override {
        board.reset();
    }
};

// Test board initialization
TEST_F(ChessBoardTest, ResetBoard) {
    EXPECT_EQ(board.get_pieces(WHITE, PAWN), 0x000000000000FF00ULL);
    EXPECT_EQ(board.get_pieces(BLACK, PAWN), 0x00FF000000000000ULL);
    EXPECT_EQ(board.get_pieces(WHITE, KING), 0x0000000000000010ULL);
    EXPECT_EQ(board.get_pieces(BLACK, KING), 0x1000000000000000ULL);
}

// Test piece occupancy
TEST_F(ChessBoardTest, IsOccupied) {
    EXPECT_TRUE(board.is_occupied(E1)); // White King
    EXPECT_TRUE(board.is_occupied(D8)); // Black Queen
    EXPECT_FALSE(board.is_occupied(E4)); // Empty square
}