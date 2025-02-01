#pragma once
#include "chessboard.hpp"
#include "movegen.hpp"
#include <string>

class Game {
private:
    ChessBoard board;

public:
    void start();
    void display_board() const;
    Move parse_input(const std::string& input) const;
};
