#pragma once
#include "chessboard.hpp"
#include "move.hpp"
#include "movegen.hpp"
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
    void play_vs_pc();
    std::pair<Square, Square> parse_input(const std::string& from, const std::string& to) const;
};
