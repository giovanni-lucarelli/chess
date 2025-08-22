// env_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <environment.hpp>
#include "game.hpp"
#include "move.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chess_py, m) {
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

    py::enum_<Square>(m, "Square")
        .value("A1", Square::A1).value("B1", Square::B1).value("C1", Square::C1).value("D1", Square::D1)
        .value("E1", Square::E1).value("F1", Square::F1).value("G1", Square::G1).value("H1", Square::H1)
        .value("A2", Square::A2).value("B2", Square::B2).value("C2", Square::C2).value("D2", Square::D2)
        .value("E2", Square::E2).value("F2", Square::F2).value("G2", Square::G2).value("H2", Square::H2)
        .value("A3", Square::A3).value("B3", Square::B3).value("C3", Square::C3).value("D3", Square::D3)
        .value("E3", Square::E3).value("F3", Square::F3).value("G3", Square::G3).value("H3", Square::H3)
        .value("A4", Square::A4).value("B4", Square::B4).value("C4", Square::C4).value("D4", Square::D4)
        .value("E4", Square::E4).value("F4", Square::F4).value("G4", Square::G4).value("H4", Square::H4)
        .value("A5", Square::A5).value("B5", Square::B5).value("C5", Square::C5).value("D5", Square::D5)
        .value("E5", Square::E5).value("F5", Square::F5).value("G5", Square::G5).value("H5", Square::H5)
        .value("A6", Square::A6).value("B6", Square::B6).value("C6", Square::C6).value("D6", Square::D6)
        .value("E6", Square::E6).value("F6", Square::F6).value("G6", Square::G6).value("H6", Square::H6)
        .value("A7", Square::A7).value("B7", Square::B7).value("C7", Square::C7).value("D7", Square::D7)
        .value("E7", Square::E7).value("F7", Square::F7).value("G7", Square::G7).value("H7", Square::H7)
        .value("A8", Square::A8).value("B8", Square::B8).value("C8", Square::C8).value("D8", Square::D8)
        .value("E8", Square::E8).value("F8", Square::F8).value("G8", Square::G8).value("H8", Square::H8)
        .value("NO_SQUARE", Square::NO_SQUARE)
        .export_values();

    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def("__repr__", [](const Move&) { return "<Move>"; })
        .def_readonly("from_square", &Move::from)
        .def_readonly("to_square", &Move::to)

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
            R"doc(Create a Move from a UCI string like "e2e4" using the context of `game`.)doc")

        .def_static("to_uci",
            [](const Move& m) {
                std::string uci = m.to_string(); // Gets "e2e4" format
                
                // Add promotion piece if applicable
                if (m.type == MoveType::PROMOTION) {
                    switch (m.promoted_to) {
                        case Piece::QUEEN:  uci += "q"; break;
                        case Piece::ROOK:   uci += "r"; break;
                        case Piece::BISHOP: uci += "b"; break;
                        case Piece::KNIGHT: uci += "n"; break;
                        default: break; // No promotion piece or invalid
                    }
                }
                
                return uci;
            },
            R"doc(Return a UCI string representation of the Move, like "e2e4" or "e7e8q".)doc");

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
        .def("parse_action_to_move", &Game::parse_action_to_move, "action"_a)
        .def("reset_from_fen", &Game::reset_from_fen, "fen"_a)
        .def("to_fen", &Game::to_fen)

        // --- Utility / state ---
        .def("is_move_legal", &Game::is_move_legal, "move"_a)
        .def("in_check", &Game::in_check)
        .def("is_checkmate", &Game::is_checkmate)
        .def("is_stalemate", &Game::is_stalemate)
        .def("is_game_over", &Game::is_game_over)
        .def("is_insufficient_material", &Game::is_insufficient_material)
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

        // Constructor: Env(const std::string& fen, double gamma=1.0, double step_penalty=0.0)
        .def(py::init<const std::string&, double, double>(),
             py::arg("fen"),
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
