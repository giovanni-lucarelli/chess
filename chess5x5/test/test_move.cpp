#include <gtest/gtest.h>
#include "move.hpp"
#include "chessboard.hpp"
#include "game.hpp"

TEST(MoveTest, NormalMove) {
    Game game;
    ChessBoard board;
    board.add_piece(WHITE, KING, D2);

    game.set_board(board);

    Move move(WHITE, KING, D2, D3, game);

    EXPECT_EQ(move.type, NORMAL);
    EXPECT_EQ(move.from, D2);
    EXPECT_EQ(move.to, D3);
    EXPECT_EQ(move.captured_piece, NO_PIECE);
}

TEST(MoveTest, CaptureMove) {
    Game game;
    ChessBoard board;
    board.add_piece(WHITE, ROOK, A1);
    board.add_piece(BLACK, KING, A5);

    game.set_board(board);

    Move move(WHITE, ROOK, A1, A5, game);

    EXPECT_EQ(move.type, CAPTURE);
    EXPECT_EQ(move.captured_piece, KING);
}

TEST(MoveTest, QueenMove) {
    Game game;
    ChessBoard board;
    board.add_piece(WHITE, QUEEN, C3);

    game.set_board(board);

    Move move(WHITE, QUEEN, C3, C4, game);

    EXPECT_EQ(move.type, NORMAL);
}

TEST(MoveTest, RookMove) {
    Game game;
    ChessBoard board;
    board.add_piece(WHITE, ROOK, E2);

    game.set_board(board);

    Move move(WHITE, ROOK, E2, E4, game);

    EXPECT_EQ(move.type, NORMAL);
}