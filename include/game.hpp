#pragma once
#include "chessboard.hpp"
#include <string>
#include <stack>


class Game {
private:
    ChessBoard board;
    std::vector<ChessBoard> board_history;
    std::stack<std::set<Square>> move_history;
    int turn;

public:
    Game();
    void play();
    std::pair<Square, Square> parse_input(const std::string& from, const std::string& to) const;
};
