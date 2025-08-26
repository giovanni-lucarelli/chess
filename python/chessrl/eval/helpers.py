from __future__ import annotations
from chessrl.env import Env, SyzygyDefender
from chessrl import chess_py as cp
import chess, chess.syzygy
from typing import Callable, Dict, Optional
from math import inf

def legal_moves_uci(fen: str):
    g = cp.Game(); g.reset_from_fen(fen)
    side = g.get_side_to_move()
    return [cp.Move.to_uci(m) for m in g.legal_moves(side)]

def optimal_moves_syzygy(tb_path: str, draw_mode: str = "all"):
    """
    fen -> set[uci] of optimal white moves, using Syzygy:
      1) maximize WDL from WHITE POV
      2) tie-break:
         - winning  (WDL=+2): minimize |DTZ|  (fastest practical win)
         - drawing  (WDL= 0): 'all' keeps all draw moves, 'min_dtz' narrows to min |DTZ|
         - losing   (WDL=-2): maximize |DTZ|  (slowest practical loss)
    """
    tb = chess.syzygy.open_tablebase(tb_path)

    def fn(fen: str) -> set[str]:
        b = chess.Board(fen)
        mates = set()
        entries = []  # (uci, wdl_white, dtz_abs)

        for mv in b.legal_moves:
            b.push(mv)

            # short-circuit: mate in 1 (side to move is Black and checkmated)
            if b.is_checkmate():
                mates.add(mv.uci())
                b.pop()
                continue

            # WDL from side-to-move (python-chess convention)
            try:
                wdl_stm = tb.probe_wdl(b)   #  +2=stm win, 0=draw, -2=stm loss
            except Exception:
                wdl_stm = None

            # Convert to WHITE POV
            if wdl_stm is None:
                wdl_white = -999  # very bad / unknown
            else:
                wdl_white = wdl_stm if b.turn == chess.WHITE else -wdl_stm

            # DTZ as distance proxy (abs)
            try:
                dtz = abs(tb.probe_dtz(b))
            except Exception:
                dtz = 0 if b.is_game_over() else 10**9

            entries.append((mv.uci(), wdl_white, dtz))
            b.pop()

        # If any mate-in-one moves were found, return them all
        if mates:
            return mates

        if not entries:
            return set()

        # Filter by best WDL (White POV)
        best_wdl = max(w for _, w, _ in entries)
        cand = [(uci, d) for (uci, w, d) in entries if w == best_wdl]

        if best_wdl == +2:            # winning → fastest |DTZ|
            best_d = min(d for _, d in cand)
            return {uci for (uci, d) in cand if d == best_d}
        elif best_wdl == 0:           # drawing
            if draw_mode == "min_dtz":
                best_d = min(d for _, d in cand)
                return {uci for (uci, d) in cand if d == best_d}
            else:  # 'all'
                return {uci for (uci, _) in cand}
        else:                          # losing → slowest |DTZ|
            best_d = max(d for _, d in cand)
            return {uci for (uci, d) in cand if d == best_d}

    return fn

# ---------- small helpers ----------

def game_from_fen(fen: str) -> cp.Game:
    g = cp.Game()
    g.reset_from_fen(fen)
    return g

def legal_moves_uci(fen: str) -> list[str]:
    g = game_from_fen(fen)
    side = g.get_side_to_move()
    return [cp.Move.to_uci(m) for m in g.legal_moves(side)]

# ---------- VI: from saved policy map (fen -> uci) ----------

def vi_move_from_policy_map(policy_map: Dict[str, Optional[str]]) -> Callable[[str, Optional[int]], str]:
    """
    Returns a move_fn that uses a precomputed greedy policy (fen -> uci).
    If fen is missing, falls back to any legal move.
    """
    def move_fn(fen: str, budget: Optional[int] = None) -> str:
        u = policy_map.get(fen)
        if u: 
            return u
        ms = legal_moves_uci(fen)
        return ms[0] if ms else ""
    return move_fn

# ---------- VI: from values V(s) with defender (2-ply greedy) ----------

def vi_move_from_values(V, tb_path):
    def move_fn(fen: str, budget=None) -> str:
        ms = legal_moves_uci(fen)
        if not ms:
            return ""
        best_u, best_val = ms[0], float("-inf")
        for u in ms:
            e = Env.from_fen(
            fen, gamma=1.0,
            defender=SyzygyDefender(tb_path), absorb_black_reply=True
            )
            sr = e.step(u)
            if sr.done:
                val = sr.reward
            else:
                val = sr.reward + V.get(e.to_fen(), -1e9)
            if val > best_val:
                best_val, best_u = val, u
        return best_u
    return move_fn

# ---------- MCTS ----------

# def mcts_move_from_instance(mcts) -> Callable[[str, Optional[int]], str]:
#     """
#     Wrap a chessrl.mcts.MCTS instance. 'budget' (if given) sets iterations.
#     """
#     def move_fn(fen: str, budget: Optional[int] = None) -> str:
#         if budget is not None:
#             try:
#                 mcts.iterations = int(budget)
#             except Exception:
#                 pass
#         g = game_from_fen(fen)
#         mv = mcts.search(g)
#         return cp.Move.to_uci(mv) if mv else ""
#     return move_fn

def mcts_move_from_instance(mcts, mode="iterations"):
    def move_fn(fen, budget=None):
        if budget is not None:
            if mode == "iterations":
                mcts.iterations = int(budget)
                mcts.seconds = 0.0
            elif mode == "seconds":
                mcts.seconds = float(budget)
        g = cp.Game(); g.reset_from_fen(fen)
        mv = mcts.search(g)
        return cp.Move.to_uci(mv) if mv else ""
    return move_fn
