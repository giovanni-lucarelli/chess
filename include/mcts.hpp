// mcts_simple.hpp
#pragma once
#include <memory>
#include <vector>
#include <random>
#include "game.hpp"
#include "move.hpp"

class MCTS_Simple {
public:
    /// seconds: wall-clock budget
    /// iterations: max number of playouts
    MCTS_Simple(double seconds = 1.0,
                std::size_t iterations = 50'000);

    /// Run search from the given position; returns best move
    Move search(const Game& rootState);

private:
    struct Node {
        Move                    fromParent;
        int                     visits   = 0;
        double                  wins     = 0;
        int                     player;            // side to move at this node
        std::vector<Move>       untried;
        std::vector<std::unique_ptr<Node>> children;
        Node*                   parent   = nullptr;
    };

    Node* select(Node* node);
    Node* expand(Node* node, Game& state);
    double simulate(Game state);
    void backprop(Node* node, double result);

    double              timeLimit;
    std::size_t         iterLimit;
    std::mt19937        rng;
};
