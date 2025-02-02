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

// Test moving a piece
TEST_F(ChessBoardTest, MovePiece) {
    board.move_piece(E2, E4); // Move white pawn
    EXPECT_FALSE(board.is_occupied(E2));
    EXPECT_TRUE(board.is_occupied(E4));
}

// Test capturing a piece
TEST_F(ChessBoardTest, CapturePiece) {
    board.move_piece(E2, E4);
    board.move_piece(D7, D5);
    board.move_piece(E4, D5); // White captures Black pawn
    EXPECT_FALSE(board.is_occupied(E4));
    EXPECT_TRUE(board.is_occupied(D5));
    auto captured_piece = board.get_piece_on_square(D5);
    EXPECT_EQ(captured_piece.first, WHITE);
    EXPECT_EQ(captured_piece.second, PAWN);
}

// Test illegal move (moving opponent's piece)
TEST_F(ChessBoardTest, IllegalMoveWrongTurn) {
    EXPECT_FALSE(board.is_move_legal(D7, D5)); // Black pawn can't move on White's turn
}

// Test legal moves
TEST_F(ChessBoardTest, LegalMove) {
    EXPECT_TRUE(board.is_move_legal(E2, E4)); // White pawn forward two squares
    EXPECT_TRUE(board.is_move_legal(G1, F3)); // Knight move
}

// Test illegal moves
TEST_F(ChessBoardTest, IllegalMove) {
    EXPECT_FALSE(board.is_move_legal(E2, E5)); // Pawns can't jump
    EXPECT_FALSE(board.is_move_legal(E1, E3)); // King can't move two squares
}
