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
#include "environment.hpp"
#include <random>

// void random_walk_roundtrip(int steps, uint64_t seed=42) {
//     std::mt19937 rng(seed);
//     Game g; g.reset_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//     for (int i=0;i<steps;++i) {
//         auto moves = g.legal_moves(g.get_side_to_move());
//         if (moves.empty()) break;
//         std::uniform_int_distribution<size_t> d(0, moves.size()-1);
//         auto m = moves[d(rng)];
//         g.do_move(m);
//     }
//     std::string fen = g.to_fen();
//     Game h; h.reset_from_fen(fen);
//     // Cheap equivalence: FENs should match (since you emit canonical stm/castling/ep fields)
//     assert(fen == h.to_fen());
// }

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

   
    // while(!game.is_game_over()){
    //     // MCTS_Simple mcts(10.0, 10000);
    //     // Move best = mcts.search(game);
    //     Move best = parallel_mcts(game, 5, 10.0, 50000);
    //     best.print();
    //     game.do_move(best);
    //     auto updated_brd = game.get_board();
    //     updated_brd.print();
    // }

    /* ------------------------------- TESTING ENV ------------------------------ */

    Env env(game, 0.99, 0.01);
    env.state().get_board().print();
    Move move = env.state().parse_move("h1", "h7");
    StepResult result = env.step(move);
    std::cout << "After move: " << std::endl;
    env.state().get_board().print();
    std::cout << "Reward: " << result.reward << ", Done: " << (result.done ? "Yes" : "No") << std::endl;
    move = env.state().parse_move("e8", "f8");
    result = env.step(move);
    std::cout << "After move: " << std::endl;
    env.state().get_board().print();
    std::cout << "Reward: " << result.reward << ", Done: " << (result.done ? "Yes" : "No") << std::endl;
    move = env.state().parse_move("d1", "d8");
    result = env.step(move);
    std::cout << "After move: " << std::endl;
    env.state().get_board().print();
    std::cout << "Reward: " << result.reward << ", Done: " << (result.done ? "Yes" : "No") << std::endl;

    return 0;
}
