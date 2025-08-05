#include <gtest/gtest.h>
#include "move.hpp"
#include "chessboard.hpp"
#include "game.hpp"

TEST(MoveTest, NormalMove) {
    Game game;
    ChessBoard board;
    board.add_piece(WHITE, PAWN, D2);

    game.set_board(board);

    Move move(WHITE, PAWN, D2, D3, game);

    EXPECT_EQ(move.type, NORMAL);
    EXPECT_EQ(move.from, D2);
    EXPECT_EQ(move.to, D3);
    EXPECT_EQ(move.captured_piece, NO_PIECE);
}

TEST(MoveTest, CaptureMove) {
    Game game;
    ChessBoard board;
    board.add_piece(WHITE, ROOK, A1);
    board.add_piece(BLACK, KNIGHT, A8);

    game.set_board(board);

    Move move(WHITE, ROOK, A1, A8, game);

    EXPECT_EQ(move.type, CAPTURE);
    EXPECT_EQ(move.captured_piece, KNIGHT);
}

TEST(MoveTest, CastlingMove) {
    Game game;
    ChessBoard board;
    board.add_piece(WHITE, KING, E1);
    board.add_piece(WHITE, ROOK, H1);

    game.set_board(board);

    Move move(WHITE, KING, E1, G1, game);

    EXPECT_EQ(move.type, CASTLING);
}

TEST(MoveTest, EnPassantMove) {
    Game game;
    game.set_en_passant_square(D6);

    ChessBoard board;
    board.add_piece(WHITE, PAWN, E5);
    board.add_piece(BLACK, PAWN, D5);

    game.set_board(board);

    Move move(WHITE, PAWN, E5, D6, game);

    EXPECT_EQ(move.type, EN_PASSANT);
}