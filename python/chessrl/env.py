from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import sys
import requests
from functools import lru_cache

# Prefer the packaged module name
try:
    from . import chess_py as cp          # packaged (wheel / editable install)
except Exception:
    # Fallbacks for in-repo dev if someone runs without installing
    try:
        import chess_py as cp             # type: ignore # plain module name in build dir
    except Exception as e:
        raise ImportError(f"Could not import chess_py module: {e}")

from chessrl.utils.plot_chess import plot_game
import matplotlib.pyplot as plt


# ---------- Optional defenders ----------------

class LichessDefender:
    """Online optimal reply via Lichess tablebase."""
    def __init__(self, session: Optional[requests.Session] = None, timeout: float = 3.0):
        self.sess = session or requests.Session()
        self.timeout = timeout

    def best_reply_uci(self, fen: str) -> str | None:
        try:
            r = self.sess.get(
                "https://tablebase.lichess.ovh/standard",
                params={"fen": fen},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            mv = data.get("moves")
            if mv:
                return mv[0]["uci"]
            else:
                return None
        except Exception:
            print(f"Warning: Lichess defender request failed for FEN: {fen}", file=sys.stderr)
            return None


class SyzygyDefender:
    """Local, offline defender using python-chess syzygy tables (maximize DTZ)."""
    def __init__(self, tb_path: str):
        import chess
        import chess.syzygy
        self.chess = chess
        try:
            self.tb = chess.syzygy.open_tablebase(tb_path)
        except Exception as e:
            raise ValueError(f"Failed to open Syzygy tablebase at '{tb_path}': {e}")

    def best_reply_uci(self, fen: str) -> str | None:
        return self._best_reply_cached(fen)

    @lru_cache(maxsize=200_000)
    def _best_reply_cached(self, fen: str) -> str | None:
        board = self.chess.Board(fen)
        best, best_score = None, float("-inf")
        for mv in board.legal_moves:
            board.push(mv)
            try:
                score = abs(self.tb.probe_dtz(board))
            except Exception:
                score = 0
                raise RuntimeError(f"Failed to probe DTZ for position {board.fen()}")
            board.pop()
            if (score > best_score and best_score!=0) or (score==0):
                best_score, best = score, mv
        return best.uci() if best else None


# ---------- Step result (Python-side) -----------------------------------------

@dataclass(slots=True)
class StepResult:
    reward: float
    done: bool
    info: dict[str, Any] | None = None


# ---------- Env ---------------------------------------------------------------

class Env:
    """
    Fast Python wrapper around chess_py.Game with optional defender absorption.

    - Accepts agent move as either a chess_py.Move or a UCI string.
    - If not terminal and side-to-move becomes Black, optionally queries a defender
      (Lichess or Syzygy) and applies Black's best reply inside the same step.
    - Rewards:
        * Terminal: return full game result (White POV: +1/0/-1), no step penalty.
        * Non-terminal: return -step_penalty.
      No gamma discount is applied (gamma kept only for interface parity).
    """

    __slots__ = (
        "game",
        "gamma",
        "step_penalty",
        "defender",
        "absorb_black_reply",
        "two_ply_cost",     
        "draw_penalty",     
        "ply",
        "exact_plies"
    )

    def __init__(
            self, 
            game: cp.Game, 
            gamma: float = 1.0, 
            step_penalty: float = 0.0, 
            defender: Any | None = None,
            absorb_black_reply: bool = True,
            two_ply_cost: float = 2.0, 
            draw_penalty: float = 1000.0,
            exact_plies=True,
            ):
        
        self.game = game
        self.gamma = gamma
        self.step_penalty = step_penalty  # (non più usato quando absorb_black_reply=True)
        self.defender = defender
        self.absorb_black_reply = absorb_black_reply
        self.two_ply_cost = two_ply_cost
        self.draw_penalty = draw_penalty
        self.ply = 0
        self.exact_plies = exact_plies


    # --- Constructors ---------------------------------------------------------

    @classmethod
    def from_fen(
        cls,
        fen: str,
        gamma: float = 1.0,
        step_penalty: float = 0.0,
        defender: Any | None = None,
        absorb_black_reply: bool = True,
        exact_plies: bool = True,
    ) -> "Env":
        g = cp.Game()
        g.reset_from_fen(fen)
        return cls(g, gamma=gamma, step_penalty=step_penalty,
                   defender=defender, absorb_black_reply=absorb_black_reply,
                   two_ply_cost=2.0, draw_penalty=1000.0, exact_plies=exact_plies)

    # --- Core step ------------------------------------------------------------

    def step(self, move_or_uci: cp.Move | str) -> StepResult:
        """
        Applica la mossa del Bianco; se richiesto, assorbe la miglior risposta del Nero.
        Ricompense:
        - non terminale dopo 2 plies   ->  -two_ply_cost
        - mate dopo mossa del Bianco   ->  0  (o -1 se exact_plies=True)
        - draw (stallo, insuff., 3x)   ->  -draw_penalty
        """
        info = {"absorbed_reply": False, "reply_uci": None}

        # --- 1) mossa del Bianco ---
        self._apply(move_or_uci)
        self.ply += 1

        # Terminal after White's move? (mate or draw)
        if self.game.is_game_over():
            reward = -1.0 if self.game.is_checkmate() else -self.draw_penalty
            return StepResult(reward=reward, done=True, info=info)

        # --- 2) assorbi la miglior risposta del Nero ---
        if self.absorb_black_reply and self.defender and self._stm_is_black():
            reply_uci = self.defender.best_reply_uci(self.to_fen())
            if reply_uci:
                try:
                    self._apply_uci(reply_uci)
                    self.ply += 1
                    info["absorbed_reply"] = True
                    info["reply_uci"] = reply_uci
                    if self.game.is_game_over():
                        # dopo la risposta del Nero, se finisce è sempre patta
                        return StepResult(reward=-self.draw_penalty, done=True, info=info)
                except Exception:
                    print(f"Warning: failed to apply defender move '{reply_uci}' for {self.to_fen()}", file=sys.stderr)
            else:
                print(f"Warning: defender returned None for {self.to_fen()}", file=sys.stderr)

        # --- 3) non terminale ---
        return StepResult(reward=-self.two_ply_cost, done=False, info=info)


    # --- Mirrors / conveniences ----------------------------------------------

    def state(self) -> cp.Game:
        return self.game

    def steps(self) -> int:
        return self.ply

    def reset_from_fen(self, fen: str) -> None:
        self.game.reset_from_fen(fen)
        self.ply = 0

    def to_fen(self) -> str:
        return self.game.to_fen()

    def is_terminal(self) -> bool:
        return self.game.is_game_over()

    def result_white_pov(self) -> float:
        return self.game.result()

    def display_state(self, save_path: str) -> None:
        # Your Board::print appears to return a string in bindings; print it if available.
        try:
            plot_game(
                self.game,
                save_path=save_path,
                title=f"side to move: {self.game.get_side_to_move()}",
                flipped=False,
                coordinates=True,
            )
            plt.show()
        except Exception:
            print(self.to_string())

    def to_string(self) -> str:
        side = "Black" if self._stm_is_black() else "White"
        try:
            board_str = self.game.get_board().print()
        except Exception:
            board_str = "(board print unavailable)"
        return (
            f"Current FEN: {self.to_fen()}\n"
            f"Current Board:\n{board_str}\n"
            f"Side to move: {side}\n"
            f"Is Game Over: {'Yes' if self.game.is_game_over() else 'No'}\n"
            f"Current Ply: {self.steps()}\n"
        )

    # --- Internal fast-paths (binding-aware) ---------------------------------

    def _stm_is_black(self) -> bool:
        return self.game.get_side_to_move() == cp.Color.BLACK

    def _apply(self, move_or_uci: cp.Move | str) -> None:
        if isinstance(move_or_uci, cp.Move):
            self.game.do_move(move_or_uci)
        elif isinstance(move_or_uci, str):
            self._apply_uci(move_or_uci)
        else:
            raise TypeError("step() expects a chess_py.Move or a UCI string.")

    def _apply_uci(self, uci: str) -> None:
        """
        Apply UCI via Move.from_uci(Game, uci). If promotion is encoded
        (e.g., 'e7e8q'), set the desired piece before calling do_move.
        """
        if len(uci) < 4:
            raise ValueError("UCI must have at least 4 characters.")

        # Handle promotion char (5th char) using Game.choose_promotion_piece
        if len(uci) >= 5:
            promo = uci[4].lower()
            piece = {
                "q": cp.Piece.QUEEN,
                "r": cp.Piece.ROOK,
                "b": cp.Piece.BISHOP,
                "n": cp.Piece.KNIGHT,
            }.get(promo)
            if piece is not None:
                self.game.choose_promotion_piece(piece)

        mv = cp.Move.from_uci(self.game, uci[:4])
        self.game.do_move(mv)
