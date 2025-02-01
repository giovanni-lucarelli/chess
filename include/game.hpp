#pragma once
#include "chessboard.hpp"
#include "movegen.hpp"
#include <string>
#include <stack>


class Game {
private:
    ChessBoard board;
    std::stack<ChessBoard> board_history; // Stack to store board states


public:
    void start();
    void display_board() const;
    Move parse_input(const std::string& input) const;
};
