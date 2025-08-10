#include <cmath>
#include <sstream>
#include "game.hpp"
#include "move.hpp"

struct StepResult { double reward; bool done; };

class Env {
public:
    Env(Game g, double gamma=1.0, double step_penalty=0.0)
      : game(std::move(g)), gamma(gamma), step_penalty(step_penalty), ply(0) {}

    // Apply a move; compute reward for *the move that just happened*
    StepResult step(const Move& m) {
        game.do_move(const_cast<Move&>(m));
        ++ply;

        if (game.is_game_over()) {
            double z = game.result();  // +1/0/-1 (White POV, independent of side-to-move)
            // If using gamma instead of step penalty:
            double r = (gamma < 1.0 && z != 0.0) ? std::copysign(std::pow(gamma, ply-1), z) : z;
            // Add step penalty if you use it:
            r -= step_penalty;
            return {r, true};
        } else {
            // non-terminal step reward (usually just step penalty)
            return {-step_penalty, false};
        }
    }

    const Game& state() const { return game; }
    int steps() const { return ply; }
    void reset_from_fen(const std::string& fen){ game.reset_from_fen(fen); ply=0; }
    std::string to_fen() const { return game.to_fen(); }
    bool is_terminal() const { return game.is_game_over(); }
    double result_white_pov() const { return game.result(); }
    void display_state() const { game.get_board().print(); }
    // add a method to print info about the full state (like python __str__ method)
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Current FEN: " << to_fen() << "\n";
        oss << "Current Board:\n";
        oss << game.get_board().print() << "\n";
        oss << "Side to move: " << (game.get_side_to_move() == WHITE ? "White" : "Black") << "\n";
        oss << "Is Game Over: " << (game.is_game_over() ? "Yes" : "No") << "\n";
        oss << "Current Ply: " << steps() << "\n";
        return oss.str();
    }

private:
    Game game;
    double gamma;
    double step_penalty;
    int ply = 0;
};
