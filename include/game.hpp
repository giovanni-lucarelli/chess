#pragma once
#include "chessboard.hpp"
#include "movegen.hpp"
#include <string>
#include <stack>


class Game {
private:
    ChessBoard board;
    std::stack<ChessBoard> board_history;
    int turn = 0;

public:
    void start1v1();
    void display_board() const;
    Move parse_input(const std::string& input) const;
    Square parse_single_input(const std::string& input) const;
};
