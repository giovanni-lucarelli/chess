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
                                   Square blackKingSq = E8)
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
    clear_and_place_kings(g, G2, H8);   // white king g2, black king h8

    g.board.add_piece(WHITE, ROOK, A1);

    auto rookMoves = g.pseudo_legal_moves(A1);
    // Horizontal (b1‑h1) + vertical (a2‑a8) = 14 squares.
    EXPECT_EQ(rookMoves.size(), 14u);

    std::vector<Square> expected{H1, A8, D1, A4};
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
    clear_and_place_kings(g, E1, H8);

    // Pin the white rook on the a‑file with a black rook on e‑file.
    g.board.add_piece(WHITE, ROOK, A1);
    g.board.add_piece(BLACK, ROOK, E8);

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
    clear_and_place_kings(g, F6, H8); // white king f6, black king h8
    g.board.add_piece(WHITE, QUEEN, G7);

    g.side_to_move = BLACK;            // it’s Black’s turn and he’s mated
    g.check_control();

    EXPECT_TRUE(g.get_check(BLACK));
    EXPECT_TRUE(g.legal_moves(BLACK).empty());
    EXPECT_TRUE(g.is_game_over());
}

TEST(GameOver, SimpleStalemate)
{
    Game g;
    clear_and_place_kings(g, F7, H8);
    g.board.add_piece(WHITE, QUEEN, G6);

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