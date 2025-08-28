from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import math, random, time

from chessrl import chess_py as cp
from chessrl.env import Env

_EPS = 1e-9

def _copy_game(g: cp.Game) -> cp.Game:
    """Copy by FEN to avoid depending on a C++ copy constructor."""
    h = cp.Game()
    h.reset_from_fen(g.to_fen())
    return h

# @dataclass
# class _Node:
#     from_parent: Optional[cp.Move] = None
#     visits: int = 0
#     wins: float = 0.0           # accumulated result from this node’s POV (player)
#     player: int = 0             # side to move at this node (cp.Color.*)
#     untried: List[cp.Move] = field(default_factory=list)
#     children: List["_Node"] = field(default_factory=list)
#     parent: Optional["_Node"] = None

@dataclass
class _Node:
    from_parent: Optional[cp.Move] = None
    visits: int = 0
    wins: float = 0.0
    player: int = 0
    untried: List[cp.Move] = field(default_factory=list)
    children: List["_Node"] = field(default_factory=list)
    parent: Optional["_Node"] = None

    # NEW: identify the position at this node (FEN after 'from_parent' applied; for the root it's the root FEN)
    state_fen: Optional[str] = None

    # (optional but handy) fast lookup from child move UCI -> child node
    # build it lazily; you can keep it None if you don't want this
    # move_map: Dict[str, "_Node"] = field(default_factory=dict)


from dataclasses import dataclass

@dataclass
class MoveStat:
    move: cp.Move
    visits: int
    pct: float           # fraction of root visits
    q_parent: float      # mean value from the root (White) POV
    uct_score: float     # UCT at the end (optional, for reference)

class MCTS:
    """
    Simple UCT MCTS over chessrl.chess_py.Game.

    By default, playouts are random using the fast C++ Game.
    Optionally, you can do rollouts through Env to 'absorb' a Black reply or
    use a defender (Lichess/Syzygy) during simulation.

    Args:
        seconds: wall-clock budget (set 0 to ignore)
        iterations: max playouts
        c_puct: exploration constant (sqrt(2) classic UCT)
        use_env_in_rollout: if True, playout uses Env; else pure Game
        gamma: discount used only for rollout return shaping
        step_penalty: forwarded to Env (only if use_env_in_rollout)
        defender: forwarded to Env (only if use_env_in_rollout)
        absorb_black_reply: forwarded to Env (only if use_env_in_rollout)
        max_playout_ply: safety cap for rollout length
        seed: rng seed for reproducibility
    """
    def __init__(
        self,
        seconds: float = 1.0,
        iterations: int = 50_000,
        c_puct: float = math.sqrt(2.0),
        use_env_in_rollout: bool = False,
        gamma: float = 0.99,
        step_penalty: float = 0.0,
        defender=None,
        absorb_black_reply: bool = True,
        max_playout_ply: int = 200,
        seed: Optional[int] = None,
        draw_value: float = -0.5,   # <— NEW: penalty for draws in rollout returns
    ):
        self.seconds = seconds
        self.iterations = iterations
        self.c_puct = c_puct
        self.use_env = use_env_in_rollout
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.defender = defender
        self.absorb_black_reply = absorb_black_reply
        self.max_playout_ply = max_playout_ply
        self.rng = random.Random(seed)
        self.draw_value = draw_value
        self._root: Optional[_Node] = None

    def _make_root_from_state(self, root_state: cp.Game) -> _Node:
        root = _Node()
        root.player = root_state.get_side_to_move()
        root.untried = root_state.legal_moves(root.player)
        root.state_fen = root_state.to_fen()
        return root

    def _promote_child_as_root(self, child: _Node) -> _Node:
        # Detach and forget the old parent reference so GC can reclaim siblings
        child.parent = None
        return child

    def _rebase_tree(self, root_state: cp.Game) -> _Node:
        """Return a node to use as root, reusing subtree stats when possible."""
        current_fen = root_state.to_fen()

        # Case 1: exact same root => reuse as-is
        if self._root is not None and self._root.state_fen == current_fen:
            return self._root

        # Case 2: position equals one of the previous root's children (one ply later)
        if self._root is not None:
            for c in self._root.children:
                if c.state_fen == current_fen:
                    self._root = self._promote_child_as_root(c)
                    return self._root

        # (Optional) You could add a shallow search here to handle two plies later
        # by scanning grandchildren; for now we rebuild if not found.

        # Case 3: build fresh root
        self._root = self._make_root_from_state(root_state)
        return self._root

    def search(self, root_state: cp.Game) -> Optional[cp.Move]:
        best, stats = self.search_with_stats(root_state, top_k=1, print_stats=False)
        return best

    def search_with_stats(self, root_state: cp.Game, top_k: int = 5, print_stats: bool = True):
        """Esegue MCTS e restituisce (best_move, lista MoveStat ordinata)."""
        # NEW: try to reuse a previous subtree that matches this position
        root = self._rebase_tree(root_state)

        if not root.untried and not root.children:
            # Ensure we have legal moves if node came from cache but was empty
            root.untried = root_state.legal_moves(root.player)

        if not root.untried and not root.children:
            return None, []

        deadline = time.perf_counter() + self.seconds if self.seconds > 0 else float("inf")

        for i in range(self.iterations):
            if time.perf_counter() >= deadline:
                break

            state = _copy_game(root_state)
            node = root

            # 1) Selection
            while not node.untried and node.children:
                node = self._select(node)
                state.do_move(node.from_parent)

            # 2) Expansion
            if node.untried:
                node = self._expand(node, state)

            # 3) Simulation
            result = self._simulate(state)

            # 4) Backprop
            self._backprop(node, result)

        # ---- calcolo best child e stats ----
        best = None
        best_q = -float("inf")
        for c in root.children:
            mean_child = c.wins / (c.visits + _EPS)  # POV del child
            parent_q = -mean_child                   # POV del parent (radice)
            if parent_q > best_q:
                best_q, best = parent_q, c

        stats = self._root_stats(root)
        # ordina per q_parent (puoi scegliere "visits" se preferisci)
        stats.sort(key=lambda s: s.q_parent, reverse=True)

        if print_stats:
            print(f"\nTop {min(top_k, len(stats))} white moves from root:")
            for i, s in enumerate(stats[:top_k], 1):
                try:
                    uci = cp.Move.to_uci(s.move) if s.move is not None else "<none>"
                except Exception:
                    uci = "<invalid>"
                print(f"{i:>2}. {uci:>6}  visits={s.visits:>6}  pct={s.pct:6.1%}  "
                    f"Q={s.q_parent:+.4f}  UCT={s.uct_score:+.4f}")

        return (best.from_parent if best else None), stats[:top_k]

    def _root_stats(self, root: _Node) -> List[MoveStat]:
        """Crea le statistiche per tutti i figli della radice (mosse del Bianco)."""
        total_visits = sum(c.visits for c in root.children) + _EPS
        ln_parent = math.log(root.visits + 1.0)
        out: List[MoveStat] = []
        for c in root.children:
            mean_child = c.wins / (c.visits + _EPS)   # valore medio dal POV del child
            q_parent = -mean_child                    # valore visto dal POV del parent (Bianco alla radice)
            uct = q_parent + self.c_puct * math.sqrt(ln_parent / (c.visits + _EPS))
            out.append(MoveStat(
                move=c.from_parent,
                visits=c.visits,
                pct=c.visits / total_visits,
                q_parent=q_parent,
                uct_score=uct,
            ))
        return out

    # ---------- MCTS internals ----------
    def _select(self, node: _Node) -> _Node:
        ln_parent = math.log(node.visits + 1.0)
        best, best_score = None, -float("inf")
        for c in node.children:
            mean_child = c.wins / (c.visits + _EPS)      # child POV
            parent_mean = -mean_child                    # parent POV
            uct = parent_mean + self.c_puct * math.sqrt(ln_parent / (c.visits + _EPS))
            if uct > best_score:
                best_score, best = uct, c
        return best

    # def _expand(self, node: _Node, state: cp.Game) -> _Node:
    #     idx = self.rng.randrange(len(node.untried))
    #     m = node.untried.pop(idx)
    #     state.do_move(m)

    #     child = _Node()
    #     child.from_parent = m
    #     child.parent = node
    #     child.player = state.get_side_to_move()
    #     child.untried = state.legal_moves(child.player)
    #     node.children.append(child)
    #     return child

    def _expand(self, node: _Node, state: cp.Game) -> _Node:
        idx = self.rng.randrange(len(node.untried))
        m = node.untried.pop(idx)
        state.do_move(m)

        child = _Node()
        child.from_parent = m
        child.parent = node
        child.player = state.get_side_to_move()
        child.untried = state.legal_moves(child.player)
        child.state_fen = state.to_fen()   # <-- NEW: remember position
        node.children.append(child)
        return child

    def _simulate(self, state: cp.Game) -> float:
        """Return terminal result from White POV (±1) with draw penalized, optionally discounted."""
        # helper per applicare il segno e la scontistica
        def _ret(z: float, ply: int) -> float:
            # z: +1 win white, 0 draw, -1 loss white
            if z > 0:
                return (self.gamma ** ply)
            if z < 0:
                return -(self.gamma ** ply)
            # draw
            return self.draw_value * (self.gamma ** ply)

        if self.use_env:
            env = Env(_copy_game(state), gamma=self.gamma,
                    defender=self.defender,
                    absorb_black_reply=self.absorb_black_reply)
            ply = 0
            while not env.is_terminal() and ply < self.max_playout_ply:
                moves = env.state().legal_moves(env.state().get_side_to_move())
                if not moves:
                    break
                env.step(moves[self.rng.randrange(len(moves))])
                ply += 1
            if env.is_terminal():
                z = env.state().result()  # +1 / 0 / -1 dal POV del Bianco
                return _ret(z, ply)
            # cutoff non terminale: nessun segnale
            return 0.0

        # Fast, pure-Game random playout
        g = _copy_game(state)
        ply = 0
        while not g.is_game_over() and ply < self.max_playout_ply:
            side = g.get_side_to_move()
            moves = g.legal_moves(side)
            if not moves:
                break
            g.do_move(moves[self.rng.randrange(len(moves))])
            ply += 1
        if g.is_game_over():
            z = g.result()  # +1 / 0 / -1 dal POV del Bianco
            return _ret(z, ply)
        return 0.0


    def _backprop(self, node: _Node, result: float) -> None:
        while node:
            node.visits += 1
            # Accumulate result from *this node's* side-to-move POV
            if node.player == cp.Color.WHITE:
                node.wins += result
            else:
                node.wins -= result
            node = node.parent
