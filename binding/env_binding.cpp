// env_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <environment.hpp>
#include "game.hpp"
#include "move.hpp"

namespace py = pybind11;


PYBIND11_MODULE(chess, m) {
    m.doc() = "Bindings for Env RL wrapper around Game";

    // --- StepResult ---
    py::class_<StepResult>(m, "StepResult")
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("done", &StepResult::done)
        .def("__repr__", [](const StepResult& s){
            return "<StepResult reward=" + std::to_string(s.reward) +
                   ", done=" + std::string(s.done ? "True" : "False") + ">";
        });

    using namespace pybind11::literals;

    py::enum_<Color>(m, "Color")
        .value("WHITE", Color::WHITE)
        .value("BLACK", Color::BLACK)
        .export_values();

    py::enum_<Piece>(m, "Piece")
        .value("PAWN",   Piece::PAWN)
        .value("KNIGHT", Piece::KNIGHT)
        .value("BISHOP", Piece::BISHOP)
        .value("ROOK",   Piece::ROOK)
        .value("QUEEN",  Piece::QUEEN)
        .value("KING",   Piece::KING)
        .export_values();

    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def("__repr__", [](const Move&) { return "<Move>"; })

        // Python-friendly constructors that *delegate to Game::parse_move*.
        // This keeps parsing in Game (context-aware), but is ergonomic from Python.
        .def_static("from_strings",
            [](const Game& g, const std::string& from, const std::string& to) {
                return g.parse_move(from, to);
            },
            "game"_a, "from"_a, "to"_a,
            R"doc(Create a Move by parsing "from","to" (like "e2","e4") in the context of `game`.)doc")

        .def_static("from_uci",
            [](const Game& g, const std::string& uci) {
                // Accept "e2e4" or "e7e8q" (promotion char ignored if Game::parse_move
                // decides promotion elsewhere; adapt if your Game::parse_move supports it)
                if (uci.size() < 4) throw std::runtime_error("UCI must be at least 4 chars");
                auto from = uci.substr(0,2);
                auto to   = uci.substr(2,2);
                // If your Game::parse_move needs promotion, extend your API accordingly.
                return g.parse_move(from, to);
            },
            "game"_a, "uci"_a,
            R"doc(Create a Move from a UCI string like "e2e4" using the context of `game`.)doc");

    py::class_<Game>(m, "Game")
        .def(py::init<>())

        // --- Getters ---
        .def("get_board", &Game::get_board) // returns by value (as declared)
        .def("get_side_to_move", &Game::get_side_to_move)
        .def("get_en_passant_square", &Game::get_en_passant_square)
        .def("get_castling_rights", &Game::get_castling_rights,
             "color"_a, "kingside"_a)
        .def("get_check", &Game::get_check, "color"_a)

        // --- Setters ---
        .def("set_side_to_move", &Game::set_side_to_move, "color"_a)
        .def("set_en_passant_square", &Game::set_en_passant_square, "sq"_a)
        .def("set_castling_rights", &Game::set_castling_rights,
             "color"_a, "kingside"_a, "value"_a)
        .def("set_board", &Game::set_board, "board"_a)
        .def("update_check", &Game::update_check)

        // --- Parse / FEN ---
        .def("parse_move", &Game::parse_move, "from"_a, "to"_a)
        .def("reset_from_fen", &Game::reset_from_fen, "fen"_a)
        .def("to_fen", &Game::to_fen)

        // --- Utility / state ---
        .def("is_move_legal", &Game::is_move_legal, "move"_a)
        .def("in_check", &Game::in_check)
        .def("is_checkmate", &Game::is_checkmate)
        .def("is_stalemate", &Game::is_stalemate)
        .def("is_game_over", &Game::is_game_over)
        .def("result", &Game::result)

        // --- Actions ---
        .def("do_move", &Game::do_move, "move"_a)
        .def("undo_move", &Game::undo_move, "move"_a)
        .def("choose_promotion_piece", &Game::choose_promotion_piece)

        // --- Move generation (overloads) ---
        .def("pseudo_legal_moves",
             py::overload_cast<Square>(&Game::pseudo_legal_moves, py::const_),
             "from"_a)
        .def("legal_moves",
             py::overload_cast<Square>(&Game::legal_moves, py::const_),
             "from"_a)
        .def("legal_moves",
             py::overload_cast<Color>(&Game::legal_moves, py::const_),
             "color"_a)

        // --- Misc ---
        .def("play", &Game::play)  // if it reads from stdin, call from a terminal
        .def("__repr__", [](const Game& g){
            return "<Game fen=\"" + g.to_fen() + "\">";
        });

    py::class_<Env>(m, "Env")
        // Constructor: Env(Game g, double gamma=1.0, double step_penalty=0.0)
        .def(py::init<Game, double, double>(),
             py::arg("game"),
             py::arg("gamma") = 1.0,
             py::arg("step_penalty") = 0.0)

        // Apply a move; returns StepResult
        .def("step", &Env::step, py::arg("move"),
             R"doc(Apply a Move and return StepResult(reward, done).)doc")

        // Read-only accessors / utilities
        .def("state", &Env::state,
             py::return_value_policy::reference_internal,
             R"doc(Return a reference to the underlying Game (tied to this Env).)doc")
        .def("steps", &Env::steps)
        .def("reset_from_fen", &Env::reset_from_fen, py::arg("fen"))
        .def("to_fen", &Env::to_fen)
        .def("is_terminal", &Env::is_terminal)
        .def("result_white_pov", &Env::result_white_pov)
        .def("display_state", &Env::display_state)
        .def("__str__", &Env::to_string)
        .def("__repr__", [](const Env& e){ return e.to_string(); });
}
