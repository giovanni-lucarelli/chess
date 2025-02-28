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

#include <gtest/gtest.h>
#include "chessboard.hpp"

// Extend the ChessBoardTest fixture with helper functions to set up custom states.
class ChessBoardCustomTest : public ::testing::Test {
protected:
    ChessBoard board;

    void SetUp() override {
        board.reset();
    }
    
    // Clear a square and add a piece to that square.
    void set_piece(Square sq, Color color, Piece p) {
        board.remove_piece(sq);
        board.add_piece(color, p, sq);
    }
};

// Test castling: remove blocking pieces for white kingside castling, then move king from E1 to G1.
TEST_F(ChessBoardCustomTest, WhiteKingsideCastling) {
    // Remove pieces between king and rook.
    board.remove_piece(F1);
    board.remove_piece(G1);
    
    // Force castling rights if necessary.
    // (Assumes that after reset, castling_rights for white remain true.)
    
    // Move white king from E1 to G1.
    board.move_piece(E1, G1);
    
    // Check that king is on G1.
    auto kingSquare = board.get_piece_on_square(G1);
    EXPECT_EQ(kingSquare.first, WHITE);
    EXPECT_EQ(kingSquare.second, KING);
    
    // Check that the rook has moved from H1 to F1.
    auto rookSquare = board.get_piece_on_square(F1);
    EXPECT_EQ(rookSquare.first, WHITE);
    EXPECT_EQ(rookSquare.second, ROOK);
    
    // Also, castling rights should be revoked.
    // (If you have a getter for castling rights, assert they are now false.)
}

// black kingside castling
TEST_F(ChessBoardCustomTest, BlackKingsideCastling) {
    // Remove pieces between king and rook.
    board.remove_piece(F8);
    board.remove_piece(G8);
    
    // Force castling rights if necessary.
    // (Assumes that after reset, castling_rights for black remain true.)
    
    // Move black king from E8 to G8.
    board.move_piece(E8, G8);
    
    // Check that king is on G8.
    auto kingSquare = board.get_piece_on_square(G8);
    EXPECT_EQ(kingSquare.first, BLACK);
    EXPECT_EQ(kingSquare.second, KING);
    
    // Check that the rook has moved from H8 to F8.
    auto rookSquare = board.get_piece_on_square(F8);
    EXPECT_EQ(rookSquare.first, BLACK);
    EXPECT_EQ(rookSquare.second, ROOK);
    
    // Also, castling rights
    // (If you have a getter for castling rights, assert they are now false.)
}

// Test promotion: set up a white pawn at E7 and move to E8, then check it promotes (to QUEEN if using default).
TEST_F(ChessBoardCustomTest, PawnPromotion) {
    // Remove any piece on E7 and E8.
    board.remove_piece(E7);
    board.remove_piece(E8);
    
    // Place a white pawn on E7.
    board.add_piece(WHITE, PAWN, E7);
    
    // Move pawn from E7 to E8 in non-interactive mode so it defaults to QUEEN.
    board.move_piece(E7, E8, false);
    
    // Get piece on E8.
    auto promoted = board.get_piece_on_square(E8);
    EXPECT_EQ(promoted.first, WHITE);
    EXPECT_EQ(promoted.second, QUEEN);
}

// Test en passant capture
TEST_F(ChessBoardCustomTest, EnPassantCapture) {
    // Build a custom en passant scenario.
    // Remove pawn obstacles.
    board.remove_piece(E2);
    board.remove_piece(D4);
    board.remove_piece(E4);
    
    // Place white pawn on E4 and black pawn on D4.
    board.add_piece(WHITE, PAWN, E4);
    board.add_piece(BLACK, PAWN, D4);
    
    // Simulate black pawn moving two squares forward (from D7 to D5) to enable en passant.
    board.remove_piece(D7);
    board.add_piece(BLACK, PAWN, D5);
    
    // Manually set en passant square to D6.
    board.set_en_passant_square(D6);
    
    // Now white pawn on E4 can capture en passant by moving to D5.
    board.move_piece(E4, D5);
    
    // After en passant, the black pawn that moved two squares should be captured.
    EXPECT_FALSE(board.is_occupied(D5) && board.get_piece_on_square(D5).first == BLACK);
    
    // Verify that the capturing pawn (white) is now on D5.
    auto epPiece = board.get_piece_on_square(D5);
    EXPECT_EQ(epPiece.first, WHITE);
    EXPECT_EQ(epPiece.second, PAWN);
}

// Test move and undo move
TEST_F(ChessBoardCustomTest, MoveAndUndo) {
    // Set up a custom state.
    set_piece(E2, WHITE, PAWN);
    set_piece(E4, WHITE, PAWN);
    
    // Move the white pawn from E2 to E4.
    board.move_piece(E2, E4);
    
    // Verify that the pawn is now on E4.
    auto movedPiece = board.get_piece_on_square(E4);
    EXPECT_EQ(movedPiece.first, WHITE);
    EXPECT_EQ(movedPiece.second, PAWN);
    
    // Undo the move.
    board.undo_move(E2, E4);
    
    // Verify that the pawn is back on E2.
    auto undonePiece = board.get_piece_on_square(E2);
    EXPECT_EQ(undonePiece.first, WHITE);
    EXPECT_EQ(undonePiece.second, PAWN);
}   
