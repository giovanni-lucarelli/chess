#include "game.hpp"
#include <iostream>

Game::Game() {
    board.reset();
}

std::pair<Square, Square> Game::parse_input(const std::string& from, const std::string& to) const {
    Square from_sq = static_cast<Square>(8 * (from[1] - '1') + (from[0] - 'a'));
    Square to_sq = static_cast<Square>(8 * (to[1] - '1') + (to[0] - 'a'));

    return {from_sq, to_sq};
}

