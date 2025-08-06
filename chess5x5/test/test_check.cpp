#include <gtest/gtest.h>
#define private public
#include "game.hpp"
#include "chessboard.hpp"
#include "types.hpp"
#undef private

// ------------------------- helpers ----------------------------- //

/// Empties the board and puts the two mandatory kings on.
static void clear_and_place_kings(Game &g,
                                   Square whiteKingSq = E1,
                                   Square blackKingSq = A5)
{
    g.board.clear();
    g.board.add_piece(WHITE, KING, whiteKingSq);
    g.board.add_piece(BLACK, KING, blackKingSq);
}

// --------------------------- tests ----------------------------- //

TEST(PseudoLegalMoves, RookFromCorner)
{
    Game g;
    // Place kings so they don't block the rook's horizontal/vertical lines.
    clear_and_place_kings(g, C2, B5);   // white king c2, black king b5

    g.board.add_piece(WHITE, ROOK, A1);

    auto rookMoves = g.pseudo_legal_moves(A1);
    // Horizontal (b1‑e1) + vertical (a2‑a5) = 7 squares.
    EXPECT_EQ(rookMoves.size(), 7u);

    std::vector<Square> expected{E1, A5, D1, A3};
    for (auto sq : expected)
    {
        bool found = std::any_of(rookMoves.begin(), rookMoves.end(),
                                 [&](const Move &m) { return m.to == sq; });
        EXPECT_TRUE(found) << "Missing square " << square_to_string(sq);
    }
}

TEST(LegalMoves, PinnedPieceCannotMove)
{
    Game g;
    clear_and_place_kings(g, E1, A5);

    // Pin the white rook on the a‑file with a black rook on e‑file.
    g.board.add_piece(WHITE, ROOK, A1);
    g.board.add_piece(BLACK, ROOK, E5);

    g.side_to_move = WHITE;
    g.check_control();

    auto pseudo = g.pseudo_legal_moves(A1);
    EXPECT_GT(pseudo.size(), 0u);            // rook has *some* moves in theory

    auto legal = g.legal_moves(A1);
    EXPECT_EQ(legal.size(), 0u)              // but none are legal because the king is in check
        << "Pinned rook should have no legal moves";
}

TEST(GameOver, SimpleCheckmate)
{
    Game g;
    clear_and_place_kings(g, C3, A5); // white king c3, black king a5
    g.board.add_piece(WHITE, QUEEN, B4);

    g.side_to_move = BLACK;            // it’s Black’s turn and he’s mated
    g.check_control();

    EXPECT_TRUE(g.get_check(BLACK));
    EXPECT_TRUE(g.legal_moves(BLACK).empty());
    EXPECT_TRUE(g.is_game_over());
}

TEST(GameOver, SimpleStalemate)
{
    Game g;
    clear_and_place_kings(g, C4, A5);
    g.board.add_piece(WHITE, QUEEN, B3);

    g.side_to_move = BLACK;            // Black to move – no legal moves but not in check
    g.check_control();

    EXPECT_FALSE(g.get_check(BLACK));
    EXPECT_TRUE(g.legal_moves(BLACK).empty());
    EXPECT_TRUE(g.is_game_over());
}

TEST(LegalMovesByColor, SumMatchesPerPieceEnumeration)
{
    Game g;
    clear_and_place_kings(g);
    g.board.add_piece(WHITE, ROOK, A1);

    g.side_to_move = WHITE;
    g.check_control();

    // Sum legal moves per‑piece and compare with the aggregated call.
    size_t sum = g.legal_moves(A1).size() + g.legal_moves(E1).size();
    EXPECT_EQ(sum, g.legal_moves(WHITE).size());
}