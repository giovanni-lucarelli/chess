# chessrl/eval/adapters.py
from typing import Dict, Optional, Callable, Any
from chessrl import chess_py as cp
from chessrl.utils.helpers import legal_moves_uci

# 1) PolicyMap adapter (use when you saved fen->uci)
class PolicyMapAgent:
    def __init__(self, policy_map: Dict[str, Optional[str]]):
        self.policy_map = policy_map
    def act(self, fen: str, *, budget: Optional[int] = None) -> str:
        u = self.policy_map.get(fen)
        if u: return u
        ms = legal_moves_uci(fen)
        return ms[0] if ms else ""

# 2) Values adapter (use when you saved fen->V and want greedy 2-ply using Env+Syzygy)
class GreedyFromVAgent:
    def __init__(self, V: Dict[str, float], env_factory: Callable[[str], Any]):
        self.V = V
        self.env_factory = env_factory
    def _two_ply_value(self, fen: str, uci: str) -> float:
        env = self.env_factory(fen)
        mv = cp.Move.from_uci(env.state(), uci)
        sr = env.step(mv)                       # absorbs best Black reply
        if sr.done:
            return 0.0 if sr.reward >= 0 else -1e6   # mate good, draw bad
        return -2.0 + self.V.get(env.to_fen(), float("-inf"))
    def act(self, fen: str, *, budget: Optional[int] = None) -> str:
        ms = legal_moves_uci(fen)
        return max(ms, key=lambda u: self._two_ply_value(fen, u)) if ms else ""

# 3) MCTS adapter (no training artifact needed)
class MCTSAgent:
    def __init__(self, mcts): self.mcts = mcts
    def act(self, fen: str, *, budget: Optional[int] = None) -> str:
        if budget is not None:
            self.mcts.iterations = int(budget)  # or map to seconds
        g = cp.Game(); g.reset_from_fen(fen)
        mv = self.mcts.search(g)
        return cp.Move.to_uci(mv) if mv else ""
