# import requests

# def get_black_move(self, fen):
#         """Query online tablebase (Lichess API)"""
#         url = f"http://tablebase.lichess.ovh/standard?fen={fen}"
        
#         try:
#             response = requests.get(url)
#             data = response.json()
            
#             if 'moves' in data and data['moves']:
#                 # Get the best move (first in the list)
#                 best_move_data = data['moves'][0]
#                 return best_move_data['uci']  # Return UCI string directly
#             return None
#         except:
#             return None


# two_ply_env.py
# import requests
# from dataclasses import dataclass
# import sys
# sys.path.insert(0, "../build")
# import chess_py  

from __future__ import annotations
import sys, pathlib, requests
from dataclasses import dataclass

_root = pathlib.Path(__file__).resolve().parents[1]
_build = _root / "build"
if str(_build) not in sys.path:
    sys.path.insert(0, str(_build))

import chess_py 

@dataclass
class StepResult:
    reward: float
    done: bool
    info: dict

# ---- Defender policies ----
class LichessDefender:
    def __init__(self, session=None):
        self.sess = session or requests.Session()

    def best_reply_uci(self, fen: str) -> str | None:
        try:
            r = self.sess.get("http://tablebase.lichess.ovh/standard", params={"fen": fen}, timeout=3)
            r.raise_for_status()
            data = r.json()
            if data.get("moves"):
                # First move is optimal; API returns UCI like "e7e8q" for promotions
                return data["moves"][0]["uci"]
        except Exception:
            pass
        return None

class SyzygyDefender:
    """Optional: local, faster, offline defender using python-chess.syzygy"""
    def __init__(self, tb_path):
        import chess
        import chess.syzygy
        self.chess = chess
        self.tb = chess.syzygy.open_tablebase(tb_path)

    def best_reply_uci(self, fen: str) -> str | None:
        board = self.chess.Board(fen)
        best = None
        best_score = float("-inf")
        for mv in board.legal_moves:
            board.push(mv)
            try:
                # Larger abs(DTZ) usually means more delay for White; defender maximizes it
                dtz = abs(self.tb.probe_dtz(board))
                score = dtz
            except Exception:
                score = 0
            board.pop()
            if score > best_score:
                best_score = score
                best = mv
        return best.uci() if best else None

# ---- 2-ply wrapper ----
class TwoPlyEnv:
    """
    Compose White move + Black best reply into one step.
    Reward scheme (recommended for shortest-mate training):
      - non-terminal 2-ply: -2
      - checkmate for White: 0
      - draw (stalemate/insufficient/etc.): -100  
    """
    def __init__(self, base_env: "chess_py.Env", defender=None,
                 step_cost_per_two_ply: float = 2.0, draw_penalty: float = 100.0):
        self.env = base_env
        self.defender = defender or LichessDefender()
        self.step_cost = step_cost_per_two_ply
        self.draw_penalty = draw_penalty

    # Convert UCI to C++ Move; if you have Move.from_uci(..), use that.
    def _uci_to_move(self, uci: str) -> "chess_py.Move":
        g = self.env.state()
        # if hasattr(chess_py.Move, "from_uci"):
        return chess_py.Move.from_uci(g, uci)  # preferred
        # from_sq, to_sq = uci[:2], uci[2:4]
        # # Promotions: your Game::choose_promotion_piece() can handle default 'Q';
        # # if you need explicit promotion, extend parse_move to accept it.
        # return g.parse_move(from_sq, to_sq)

    def step(self, white_uci: str) -> StepResult:
        # 1) White plays
        sr_w = self.env.step(self._uci_to_move(white_uci))
        if sr_w.done:
            # Map your engine’s terminal scores to the shortest-mate objective
            # Your Game::result currently returns ±1000 for mate and -1000 for draw.
            r = 0.0 if sr_w.reward > 0 else -self.draw_penalty
            return StepResult(reward=r, done=True, info={"who":"white", "white_uci":white_uci})

        # 2) Black best reply from tablebase
        fen = self.env.to_fen()
        black_uci = self.defender.best_reply_uci(fen)

        if black_uci is None:
            # pick any legal black move 
            legal_b = self.env.state().legal_moves(chess_py.Color.BLACK)
            if not legal_b:
                # No legal moves – treat as terminal per C++ state; re-check:
                return StepResult(reward=-self.draw_penalty, done=True, info={"who":"black"})
            # If legal_moves returns Move objects, convert one back to UCI
            black_uci = getattr(legal_b[0], "uci", None) or str(legal_b[0])

        sr_b = self.env.step(self._uci_to_move(black_uci))
        if sr_b.done:
            # If game ended after Black’s reply, it must be a draw (stalemate/insufficient),
            # because Black has only a King and cannot mate White.
            return StepResult(reward=-self.draw_penalty, done=True,
                              info={"who":"black", "white_uci":white_uci, "black_uci":black_uci})

        # 3) Non-terminal: charge the 2-ply cost
        return StepResult(reward=-self.step_cost, done=False,
                          info={"white_uci":white_uci, "black_uci":black_uci})
