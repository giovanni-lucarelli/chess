#include "game.hpp"
#include "chessboard.hpp"
#include "mcts.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <assert.h>
#include "move.hpp"

// Simple O(n²) vote‐tally, needs only operator==
static Move vote_best(const std::vector<Move>& votes) {
    std::vector<Move> unique;
    std::vector<int>   count;

    for (auto const& m : votes) {
        auto it = std::find(unique.begin(), unique.end(), m);
        if (it == unique.end()) {
            unique.push_back(m);
            count.push_back(1);
        } else {
            count[it - unique.begin()]++;
        }
    }
    auto best_i = std::distance(
        count.begin(),
        std::max_element(count.begin(), count.end())
    );
    return unique[best_i];
}

/// Run `num_threads` separate MCTS searches (each with its own RNG & tree),
/// then pick the move most often returned.
inline Move parallel_mcts(const Game& rootState,
                          int num_threads,
                          double seconds_per_thread = 0.5,
                          size_t its_per_thread  = 5000)
{
    std::vector<Move> votes(num_threads);
    std::vector<std::thread> pool;
    pool.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        pool.emplace_back([&, i]() {
            MCTS_Simple mcts(seconds_per_thread, its_per_thread);
            votes[i] = mcts.search(rootState);
        });
    }
    for (auto &t : pool) t.join();

    return vote_best(votes);
}

int main() {
    ChessBoard board;
    
    board.add_piece(BLACK, KING, E8);
    board.add_piece(WHITE, KING, E1);
    board.add_piece(WHITE, QUEEN, D1);
    // board.add_piece(WHITE, PAWN, A1);
    board.add_piece(WHITE, ROOK, H1);

    
    board.print();
    
    Game game;
    game.set_board(board);
    game.set_castling_rights(WHITE,true, false);
    game.set_castling_rights(WHITE,false, false);
    game.set_castling_rights(BLACK,true, false);
    game.set_castling_rights(BLACK,false, false);

    
    while(!game.is_game_over()){
        MCTS_Simple mcts(10.0, 10000);
        Move best = mcts.search(game);
        // Move best = parallel_mcts(game, 5, 10.0, 50000);
        best.print();
        game.do_move(best);
        auto updated_brd = game.get_board();
        updated_brd.print();
    }


    /* ---------------------------------- TEST ---------------------------------- */
    // auto moves = game.legal_moves(game.get_side_to_move());
    // //print moves
    // for (auto& m : moves) {
    //     m.print();
    // }



    // auto mv = game.parse_move(std::string("a7"),std::string("g7"));
    // game.do_move(mv);

    // std::cout << "After move: " << std::endl;
    // game.get_board().print();

    // std::cout << "Is game over? " << (game.is_game_over() ? "Yes" : "No") << std::endl;
    // std::cout << "Result: " << game.result() << std::endl;

    return 0;
}
