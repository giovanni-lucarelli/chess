#pragma once
#include "chessboard.hpp"
#include <string>
#include <stack>


class Game {
private:
    

public:

    Game();
    std::pair<Square, Square> parse_input(const std::string& from, const std::string& to) const;

    ChessBoard board;
    std::vector<ChessBoard> board_history;
    std::stack<std::set<Square>> move_history;
};
