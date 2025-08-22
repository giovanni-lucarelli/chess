#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, logging
from chessrl.chess_py import Game, Move, Color
from chessrl.algorithms.mcts import MCTS

def main():
    ap = argparse.ArgumentParser(description="MCTS self-play data generator.")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--out", type=str, default="output/mcts_selfplay.jsonl")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("mcts_selfplay")

    mcts = MCTS(seconds=args.seconds, iterations=args.iterations)
    with open(args.out, "w") as f:
        for gi in range(args.games):
            g = Game()
            # Set start FEN if your Game requires it, else default start is fine
            ply = 0
            while not g.is_game_over() and ply < args.max_plies:
                fen = g.to_fen()
                move = mcts.search(g)
                if move is None:
                    break
                uci = Move.to_uci(move)
                f.write(json.dumps({"fen": fen, "move": uci, "side": int(g.get_side_to_move())}) + "\n")
                g.do_move(move)
                ply += 1
            log.info("Game %d finished. result=%s", gi+1, g.result() if g.is_game_over() else "NA")
    log.info("Wrote %s", args.out)

if __name__ == "__main__":
    main()
