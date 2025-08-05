#include <gtest/gtest.h>
#include "game.hpp"
#include "move.hpp"

// Test initial Game state
TEST(GameTest, InitialState) {
    Game game;

    EXPECT_EQ(game.get_side_to_move(), WHITE);
    EXPECT_EQ(game.get_en_passant_square(), NO_SQUARE);

    EXPECT_TRUE(game.get_castling_rights(WHITE, true));  // Kingside white
    EXPECT_TRUE(game.get_castling_rights(WHITE, false)); // Queenside white
    EXPECT_TRUE(game.get_castling_rights(BLACK, true));  // Kingside black
    EXPECT_TRUE(game.get_castling_rights(BLACK, false)); // Queenside black

    EXPECT_FALSE(game.get_check(WHITE));
    EXPECT_FALSE(game.get_check(BLACK));
}

// Test side-to-move setter/getter
TEST(GameTest, SetSideToMove) {
    Game game;
    game.set_side_to_move(BLACK);
    EXPECT_EQ(game.get_side_to_move(), BLACK);
}

// Test en passant square setter/getter
TEST(GameTest, EnPassantSquare) {
    Game game;
    game.set_en_passant_square(E6);
    EXPECT_EQ(game.get_en_passant_square(), E6);
}


// Test castling rights setter/getter
TEST(GameTest, CastlingRights) {
    Game game;

    game.set_castling_rights(WHITE, true, false);
    EXPECT_FALSE(game.get_castling_rights(WHITE, true));
    EXPECT_TRUE(game.get_castling_rights(WHITE, false));  // Queenside still true

    game.set_castling_rights(BLACK, false, false);
    EXPECT_FALSE(game.get_castling_rights(BLACK, false));
}

// // Test promotion choice
// TEST(GameTest, PromotionChoice) {
//     Game game;
//     Piece promoted = game.choose_promotion_piece();
//     EXPECT_EQ(promoted, QUEEN); 
// }

// Test set/get board
TEST(GameTest, SetBoard) {
    Game game;
    ChessBoard board;
    board.add_piece(WHITE, KNIGHT, B1);

    game.set_board(board);

    ChessBoard new_board = game.get_board();
    EXPECT_TRUE(new_board.is_occupied(B1));
    EXPECT_EQ(new_board.get_piece_on_square(B1).second, KNIGHT);
}

TEST(GameTest, DoMoveNormalAndUndo) {
    Game game;
    ChessBoard board;

    // Place a white pawn on E2
    board.add_piece(WHITE, PAWN, E2);
    game.set_board(board);

    // Create a Move E2 -> E4
    Move move(WHITE, PAWN, E2, E4, game);

    // Perform the move
    game.do_move(move);

    ChessBoard after_board = game.get_board();
    EXPECT_FALSE(after_board.is_occupied(E2));
    EXPECT_TRUE(after_board.is_occupied(E4));
    EXPECT_EQ(after_board.get_piece_on_square(E4).second, PAWN);

    // Undo the move
    game.undo_move(move);

    ChessBoard undo_board = game.get_board();
    EXPECT_TRUE(undo_board.is_occupied(E2));
    EXPECT_FALSE(undo_board.is_occupied(E4));
}

TEST(GameTest, DoCaptureAndUndo) {
    Game game;
    ChessBoard board;

    // Place white rook on A1 and black knight on A8
    board.add_piece(WHITE, ROOK, A1);
    board.add_piece(BLACK, KNIGHT, A8);
    game.set_board(board);

    // Create a capture move A1 -> A8
    Move move(WHITE, ROOK, A1, A8, game);

    // Do capture
    game.do_move(move);

    ChessBoard after_board = game.get_board();
    EXPECT_FALSE(after_board.is_occupied(A1));
    EXPECT_TRUE(after_board.is_occupied(A8));
    EXPECT_EQ(after_board.get_piece_on_square(A8).second, ROOK);
    EXPECT_EQ(after_board.get_piece_on_square(A8).first, WHITE);

    // Undo capture
    game.undo_move(move);

    ChessBoard undo_board = game.get_board();
    EXPECT_TRUE(undo_board.is_occupied(A1));
    EXPECT_TRUE(undo_board.is_occupied(A8));
    EXPECT_EQ(undo_board.get_piece_on_square(A8).second, KNIGHT);
    EXPECT_EQ(undo_board.get_piece_on_square(A8).first, BLACK);
}

// TEST(GameTest, DoPromotionAndUndo) {
//     Game game;
//     ChessBoard board;

//     // Place white pawn on A7
//     board.add_piece(WHITE, PAWN, A7);
//     game.set_board(board);

//     // Create a promotion move A7 -> A8
//     Move move(WHITE, PAWN, A7, A8, game);

//     // Force promotion to Queen
//     move.type = PROMOTION;
//     move.promoted_to = QUEEN;

//     // Do promotion
//     game.do_move(move);

//     ChessBoard after_board = game.get_board();
//     EXPECT_FALSE(after_board.is_occupied(A7));
//     EXPECT_TRUE(after_board.is_occupied(A8));
//     EXPECT_EQ(after_board.get_piece_on_square(A8).second, QUEEN);

//     // Undo promotion
//     game.undo_move(move);

//     ChessBoard undo_board = game.get_board();
//     EXPECT_TRUE(undo_board.is_occupied(A7));
//     EXPECT_FALSE(undo_board.is_occupied(A8));
// }

TEST(GameTest, DoEnPassantAndUndo) {
    Game game;
    ChessBoard board;

    // Place white pawn on E5 and black pawn on D5
    board.add_piece(WHITE, PAWN, E5);
    board.add_piece(BLACK, PAWN, D5);

    game.set_board(board);
    game.set_en_passant_square(D6);  // Set en passant target square

    // Create en passant move E5 -> D6
    Move move(WHITE, PAWN, E5, D6, game);

    EXPECT_EQ(move.type, EN_PASSANT);

    // Do en passant capture
    game.do_move(move);

    ChessBoard after_board = game.get_board();
    EXPECT_FALSE(after_board.is_occupied(E5));
    EXPECT_TRUE(after_board.is_occupied(D6));
    EXPECT_FALSE(after_board.is_occupied(D5));  // Captured pawn removed

    // Undo en passant
    game.undo_move(move);

    ChessBoard undo_board = game.get_board();
    EXPECT_TRUE(undo_board.is_occupied(E5));
    EXPECT_TRUE(undo_board.is_occupied(D5));
    EXPECT_FALSE(undo_board.is_occupied(D6));
}