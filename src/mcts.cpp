// mcts_simple.cpp
#include "mcts.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <iomanip>

using Clock = std::chrono::steady_clock;

MCTS_Simple::MCTS_Simple(double seconds, std::size_t iterations)
  : timeLimit(seconds),
    iterLimit(iterations),
    rng(std::random_device{}())
{}

Move MCTS_Simple::search(const Game& rootState) {
    Node root;
    root.player = rootState.get_side_to_move();
    root.untried = rootState.legal_moves(rootState.get_side_to_move());

    auto deadline = Clock::now() + 
                    std::chrono::milliseconds(int(timeLimit * 1000));

    for (std::size_t i = 0;
         i < iterLimit && Clock::now() < deadline;
         ++i)
    {
        Game state = rootState;     // copy
        Node* node = &root;

        // 1) Selection
        while (node->untried.empty() && !node->children.empty()) {
            node = select(node);
            state.do_move(node->fromParent);
        }

        // 2) Expansion
        if (!node->untried.empty()) {
            node = expand(node, state);
        }

        // 3) Simulation
        double result = simulate(state);

        // 4) Backprop
        backprop(node, result);
    }

    

    std::cout << "\n--- MCTS Search Diag ---" << std::endl;
    std::cout << "Root visits: " << root.visits << "\n" << std::endl;
    std::cout << std::left
            << std::setw(12) << "Move"
            << std::setw(10) << "Visits"
            << std::setw(18) << "Raw Wins Value"
            << std::setw(15) << "Parent POV Score" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    // Sort children by visits to see the most explored moves
    std::sort(root.children.begin(), root.children.end(), [](const auto& a, const auto& b) {
        return a->visits > b->visits;
    });

    for (const auto& child : root.children) {
        if (child->visits == 0) continue;
        // Score from the parent's (root's) perspective
        double parent_pov_score = -(child->wins / child->visits);
        std::cout << std::left
                << std::setw(12) << child->fromParent.to_string()
                << std::setw(10) << child->visits
                << std::setw(18) << child->wins
                << std::setw(15) << parent_pov_score << std::endl;
    }
    std::cout << "---------------------------------------------------\n" << std::endl;


    auto best = std::max_element(
    root.children.begin(), root.children.end(),
    [](const auto& a, const auto& b){
        double qa = -(a->wins / (a->visits + 1e-9)); // convert child POV -> parent POV
        double qb = -(b->wins / (b->visits + 1e-9));
        return qa < qb;
    });

    return best != root.children.end() ? (*best)->fromParent : Move();
}

MCTS_Simple::Node* MCTS_Simple::select(Node* node) {
    Node* best = nullptr;
    double bestUct = -std::numeric_limits<double>::infinity();

    for (auto& c : node->children) {
        // child->wins is from CHILD’s pov; parent wants the opposite
        double mean = -(c->wins / (c->visits + 1e-9));
        double uct  = mean + std::sqrt(2 * std::log(node->visits + 1) / (c->visits + 1e-9));
        if (uct > bestUct) { bestUct = uct; best = c.get(); }
    }
    return best;
}

// MCTS_Simple::Node* MCTS_Simple::expand(Node* node, Game& state) {
//     if (state.is_game_over()) {
//        backprop(node, state.result());
//     return nullptr; // skip simulate()
//     }
//     std::uniform_int_distribution<std::size_t> dist(0, node->untried.size()-1);
//     std::size_t idx = dist(rng);

//     Move m = node->untried[idx];
//     node->untried.erase(node->untried.begin() + idx);
//     state.do_move(m);

//     auto child = std::make_unique<Node>();
//     child->fromParent = m;
//     child->parent     = node;
//     child->player     = state.get_side_to_move();
//     child->untried    = state.legal_moves(state.get_side_to_move());

//     Node* ptr = child.get();
//     node->children.push_back(std::move(child));
//     return ptr;
// }


MCTS_Simple::Node* MCTS_Simple::expand(Node* node, Game& state) {
    // This function is only called if node->untried is not empty,
    // so the state is guaranteed not to be terminal yet.
    // The erroneous game-over check has been removed.

    std::uniform_int_distribution<std::size_t> dist(0, node->untried.size() - 1);
    std::size_t idx = dist(rng);

    Move m = node->untried[idx];
    node->untried.erase(node->untried.begin() + idx);
    state.do_move(m); // The state might become terminal *after* this move.

    // Create the new child node.
    auto child = std::make_unique<Node>();
    child->fromParent = m;
    child->parent     = node;
    child->player     = state.get_side_to_move();
    // legal_moves() will correctly return an empty vector if the game is now over.
    child->untried    = state.legal_moves(state.get_side_to_move());

    Node* ptr = child.get();
    node->children.push_back(std::move(child));
    return ptr; // Return the new node to the search loop.
}

// double MCTS_Simple::simulate(Game state) {
//     constexpr int MAX_PLAYOUT_PLY = 200;   // cap at 200 half-moves
//     int ply = 0;
//     while (!state.is_game_over() && ply < MAX_PLAYOUT_PLY) {
//         auto moves = state.legal_moves(state.get_side_to_move());
//         if (moves.empty()) break;
//         std::uniform_int_distribution<std::size_t> d(0, moves.size()-1);
//         state.do_move(moves[d(rng)]);
//         ++ply;
//     }
//     // if we hit the depth cap without terminal, treat as draw
//     if (!state.is_game_over()) 
//         return 0.0;
    
//     return state.result();
// }


double MCTS_Simple::simulate(Game state) { // state is a copy, we can modify it freely
    constexpr int MAX_PLAYOUT_PLY = 200;
    int ply = 0;

    // The loop condition now correctly uses your is_game_over() method
    while (!state.is_game_over() && ply < MAX_PLAYOUT_PLY) {
        auto moves = state.legal_moves(state.get_side_to_move());

        // If no legal moves exist, the game is over.
        // We can simply break, as the while condition will also catch this.
        if (moves.empty()) {
            break;
        }

        // Select and perform a random move
        std::uniform_int_distribution<std::size_t> d(0, moves.size() - 1);
        Move random_move = moves[d(rng)];
        state.do_move(random_move);
        ++ply;
    }

    // Get the result from your Game class's logic.
    // It will correctly return +1, -1, or 0.
    double base_result = state.result();

    if (base_result == 0.0) {
        return 0.0; // No discount for a draw
    }

    // --- URGENCY BIAS ---
    // The core logic to reward faster wins remains the same.
    double discount = (double)ply / (double)(MAX_PLAYOUT_PLY + 1);
    return base_result * (1.0 - discount);
}

void MCTS_Simple::backprop(Node* node, double result) {
    while (node) {
        node->visits++;
        node->wins += (node->player == WHITE ? result : -result);
        node = node->parent;
    }
}