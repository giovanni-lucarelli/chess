#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, logging
from typing import Optional
import matplotlib.pyplot as plt

from chessrl.chess_py import Game, Move, Color
from chessrl.algorithms.mcts import MCTS
from chessrl.env import SyzygyDefender, LichessDefender
from chessrl.utils.plot_chess import plot_game

def make_defender(kind: Optional[str], tb_path: Optional[str], lichess_timeout: float = 3.0):
    if not kind or kind.lower() == "none":
        return None
    if kind.lower() == "syzygy":
        if not tb_path:
            raise ValueError("--tb-path is required for --defender syzygy")
        return SyzygyDefender(tb_path)
    if kind.lower() == "lichess":
        return LichessDefender(timeout=lichess_timeout)
    raise ValueError(f"Unknown defender kind: {kind}")

def best_reply_from_defender(defender, fen: str) -> Optional[str]:
    if defender is None:
        return None
    try:
        return defender.best_reply_uci(fen)
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser(description="Simulate a game using MCTS.")
    p.add_argument("--fen", type=str, default=None, help="Start FEN (default: engine startpos).")
    p.add_argument("--seconds", type=float, default=0.5, help="Time per MCTS search.")
    p.add_argument("--iterations", type=int, default=20000, help="Max playouts per search.")
    p.add_argument("--max-plies", type=int, default=200, help="Game ply cap.")
    p.add_argument("--use-env-rollout", action="store_true", help="Use Env during rollouts.")
    p.add_argument("--defender", type=str, default="none", choices=["none","syzygy","lichess"],
                   help="If set, Black uses this defender; otherwise both sides use MCTS.")
    p.add_argument("--tb-path", type=str, default=None, help="Path to Syzygy tablebases.")
    p.add_argument("--absorb-black-reply", action="store_true",
                   help="If using Env rollouts, absorb black reply during simulation.")
    p.add_argument("--plot", action="store_true", help="Save+show board images each ply.")
    p.add_argument("--plots-dir", type=str, default="output/plots", help="Where to save plots.")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("mcts_simulate")

    os.makedirs(args.plots_dir, exist_ok=True)

    g = Game()
    if args.fen:
        g.reset_from_fen(args.fen)
    else:
        g.reset_from_fen("startpos") if hasattr(g, "reset_from_fen") else None  # keep generic

    defender = make_defender(args.defender, args.tb_path)

    # Same MCTS config used for both sides (unless defender is used for Black)
    mcts = MCTS(seconds=args.seconds,
                iterations=args.iterations,
                use_env_in_rollout=args.use_env_rollout,
                absorb_black_reply=args.absorb_black_reply)

    ply = 0
    if args.plot:
        plot_game(g, save_path=os.path.join(args.plots_dir, f"turn_{ply}.png"),
                  title=f"Turn {ply}")
        plt.show()

    while not g.is_game_over() and ply < args.max_plies:
        side = g.get_side_to_move()
        move = None

        if side == Color.BLACK and defender is not None:
            # Black uses defender (Syzygy/Lichess)
            reply = best_reply_from_defender(defender, g.to_fen())
            if reply:
                move = Move.from_uci(g, reply)
        if move is None:
            # Use MCTS for whoever didn't get a defender move
            move = mcts.search(g)
        if move is None:
            log.info("No legal move found; stopping.")
            break

        g.do_move(move)
        ply += 1

        if args.plot:
            plot_game(g, save_path=os.path.join(args.plots_dir, f"turn_{ply}.png"),
                      title=f"Turn {ply}")
            plt.show()

    log.info("Game finished. is_game_over=%s, result=%.1f", g.is_game_over(), g.result() if g.is_game_over() else 0.0)

if __name__ == "__main__":
    main()
