# utils/helpers.py
from chessrl.env import Env, SyzygyDefender
from chessrl import chess_py as cp
import chess, chess.syzygy

def make_env_factory(tb_path: str):
    defender = SyzygyDefender(tb_path)            # reuse one instance
    def factory(fen: str) -> Env:
        return Env.from_fen(
            fen, gamma=1.0, step_penalty=0.0,
            defender=defender, absorb_black_reply=True
        )
    return factory

def legal_moves_uci(fen: str):
    g = cp.Game(); g.reset_from_fen(fen)
    side = g.get_side_to_move()
    return [cp.Move.to_uci(m) for m in g.legal_moves(side)]

def optimal_moves_syzygy(tb_path: str):
    tb = chess.syzygy.open_tablebase(tb_path)
    def fn(fen: str) -> set[str]:
        board = chess.Board(fen)
        # maximize WDL, then minimize abs(DTZ)
        best_wdl = -9
        best_abs_dtz = None
        best = set()
        for mv in board.legal_moves:
            board.push(mv)
            try:
                wdl = tb.probe_wdl(board)               # +2 win, 0 draw, -2 loss
                dtz = abs(tb.probe_dtz(board))          # fallback to abs distance
            except Exception:
                wdl, dtz = -9, 10**9
            board.pop()
            key = (wdl, -dtz)                           # higher is better
            if key > (best_wdl, best_abs_dtz if best_abs_dtz is not None else float("-inf")):
                best_wdl, best_abs_dtz, best = wdl, -dtz, {mv.uci()}
            elif wdl == best_wdl and best_abs_dtz is not None and -dtz == best_abs_dtz:
                best.add(mv.uci())
        return best
    return fn

def dtm_from_dtz_map(dtz_map: dict[str,int]):
    # For KRK/KQK/KBBK, DTM ≈ abs(DTZ)
    return lambda fen: int(abs(dtz_map[fen]))
