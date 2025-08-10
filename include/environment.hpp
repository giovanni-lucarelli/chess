#include <cmath>
#include "game.hpp"
#include "move.hpp"

struct StepResult { double reward; bool done; };

class Env {
public:
    Env(Game g, double gamma=1.0, double step_penalty=0.0)
      : game(std::move(g)), gamma(gamma), step_penalty(step_penalty), ply(0) {}

    // Apply a move; compute reward for *the move that just happened*
    StepResult step(const Move& m) {
        game.do_move(const_cast<Move&>(m));  // your API mutates Move; adapt if needed
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

private:
    Game game;
    double gamma;
    double step_penalty;
    int ply = 0;
};
