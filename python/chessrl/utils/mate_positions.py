from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import itertools, random, functools

from chessrl import chess_py as cp

# ----------- helpers: piece combo & FEN building -----------

def parse_combo(combo: str) -> Tuple[List[str], List[str]]:
    s = combo.replace("vs","v").replace("VS","v").replace("V","v").strip()
    if "v" in s:
        L, R = s.split("v", 1)
        W = [c.upper() for c in L if c.lower() in "kqrbnp"]
        B = [c.lower() for c in R if c.lower() in "kqrbnp"]
    else:
        ks = [i for i,c in enumerate(s) if c.lower()=="k"]
        if len(ks)!=2:
            raise ValueError("Use like 'KRvK', 'KQvK', 'KRvKB'.")
        L, R = s[:ks[1]], s[ks[1]:]
        W = [c.upper() for c in L if c.lower() in "kqrbnp"]
        B = [c.lower() for c in R if c.lower() in "kqrbnp"]
    if W.count("K")!=1 or B.count("k")!=1:
        raise ValueError("Each side must have one king.")
    return W, B

def _sq_to_rf(sq: int) -> Tuple[int,int]: return divmod(sq, 8)

def _kings_adjacent(a: int, b: int) -> bool:
    ar,af = _sq_to_rf(a); br,bf = _sq_to_rf(b)
    return max(abs(ar-br), abs(af-bf)) <= 1

def _bad_pawn(p: str, sq: int) -> bool:
    r,_ = _sq_to_rf(sq)
    return (p=="P" and r==7) or (p=="p" and r==0)

def _build_fen(white: List[Tuple[str,int]], black: List[Tuple[str,int]], side: str) -> str:
    board = [["" for _ in range(8)] for _ in range(8)]
    for p,sq in white:
        r,f = _sq_to_rf(sq); board[7-r][f] = p
    for p,sq in black:
        r,f = _sq_to_rf(sq); board[7-r][f] = p
    rows = []
    for r in range(8):
        run = 0; row = ""
        for f in range(8):
            s = board[r][f]
            if not s: run += 1
            else:
                if run: row += str(run); run = 0
                row += s
        if run: row += str(run)
        rows.append(row or "8")
    return f"{'/'.join(rows)} {side} - - 0 1"

def _copy_game(g: cp.Game) -> cp.Game:
    h = cp.Game(); h.reset_from_fen(g.to_fen()); return h

# ----------- exact mate-in-N solver (quantified recursion) -----------

@functools.lru_cache(maxsize=200_000)
def _can_mate_in_f_cached(fen: str, attacker: int, k: int) -> bool:
    g = cp.Game(); g.reset_from_fen(fen)
    return _can_mate_in_f(g, attacker, k)

def _can_mate_in_f(g: cp.Game, attacker: int, k: int) -> bool:
    # Terminal checks
    if g.is_checkmate():
        # If side-to-move is checkmated, then the *other* side delivered mate on previous ply.
        return g.get_side_to_move() != attacker  # True if attacker has already mated
    if k == 0:
        return False

    side = g.get_side_to_move()
    moves = g.legal_moves(side)

    if side == attacker:
        # Existential: attacker needs ONE move that keeps a forced mate within k-1
        for mv in moves:
            h = _copy_game(g); h.do_move(mv)
            if _can_mate_in_f_cached(h.to_fen(), attacker, k-1):
                return True
        return False
    else:
        # Universal: defender needs to AVOID mate; attacker must mate against ALL replies
        if not moves:
            # stalemate (draw) -> not a mate
            return False
        for mv in moves:
            h = _copy_game(g); h.do_move(mv)
            if not _can_mate_in_f_cached(h.to_fen(), attacker, k):
                return False
        return True

def is_mate_in_exact(g: cp.Game, attacker: int, n: int) -> bool:
    """Exact mate in n attacker moves (not <= n)."""
    if n <= 0:
        # exact 0 only if position is already checkmate against defender
        return g.is_checkmate() and g.get_side_to_move() != attacker
    fen = g.to_fen()
    return _can_mate_in_f_cached(fen, attacker, n) and not _can_mate_in_f_cached(fen, attacker, n-1)

# ----------- smart sampler (fast for KRvK / KQvK) -----------

def _smart_squares_for(combo: str, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Heuristic placer: bias kings near edges/corners to increase wins."""
    W, B = parse_combo(combo)
    total = len(W) + len(B)
    squares = list(range(64))
    wiK, biK = W.index("K"), B.index("k")

    # Bias: place black king near corner (for faster mates)
    corners = [0,7,56,63]  # a1,h1,a8,h8 in 0..63
    bK = rng.choice(corners)
    squares.remove(bK)

    # place white king not adjacent
    ok = [s for s in squares if not _kings_adjacent(s, bK)]
    wK = rng.choice(ok); squares.remove(wK)

    # remaining pieces anywhere legal (avoid illegal pawn ranks)
    def draw_piece_slots(pieces: List[str], prefilled_idx: int, prefilled_sq: int) -> List[int]:
        out = [None]*len(pieces)
        out[prefilled_idx] = prefilled_sq
        for i, p in enumerate(pieces):
            if out[i] is not None: continue
            choices = [s for s in squares if not (p.lower()=="p" and _bad_pawn(p, s))]
            s = rng.choice(choices)
            out[i] = s; squares.remove(s)
        return out

    w_sq = draw_piece_slots(W, wiK, wK)
    b_sq = draw_piece_slots(B, biK, bK)
    return w_sq, b_sq

def generate_endgames_offline(
    combo: str,
    mate_in: int,
    side_to_move: str = "w",
    *,
    max_positions: int = 50,
    max_tries: int = 100_000,
    seed: int = 0xC0FFEE,
) -> List[str]:
    """
    No-internet generator: sample placements, then filter by exact mate-in-N
    using your C++ core + memoized solver. Works very well for KRvK, KQvK;
    for larger material, raise 'max_tries' or reduce 'mate_in'.
    """
    rng = random.Random(seed)
    W, B = parse_combo(combo)
    out: List[str] = []
    seen = set()
    attacker = cp.Color.WHITE if side_to_move == "w" else cp.Color.BLACK

    for _ in range(max_tries):
        w_sq, b_sq = _smart_squares_for(combo, rng)
        if _kings_adjacent(w_sq[W.index("K")], b_sq[B.index("k")]):  # belt & suspenders
            continue
        if any(_bad_pawn(p,s) for p,s in zip(W, w_sq)) or any(_bad_pawn(p,s) for p,s in zip(B, b_sq)):
            continue

        fen = _build_fen(list(zip(W, w_sq)), list(zip(B, b_sq)), side_to_move)
        if fen in seen:
            continue
        seen.add(fen)

        g = cp.Game()
        try:
            g.reset_from_fen(fen)
        except Exception:
            continue

        # filter by exact distance
        if is_mate_in_exact(g, attacker, mate_in):
            out.append(fen)
            if len(out) >= max_positions:
                break

    return out

# --- optional validity check with python-chess (if available) ---
try:
    import chess
    _HAVE_CHESS = True
except Exception:
    _HAVE_CHESS = False

def _fen_is_valid(fen: str) -> bool:
    if not _HAVE_CHESS:
        return True  # skip strict checks if python-chess isn't installed
    try:
        b = chess.Board(fen)
        return b.is_valid()
    except Exception:
        return False

# --- uniform sampler (no bias) ---
def _uniform_squares_for(combo: str, rng: random.Random) -> Tuple[List[int], List[int]]:
    W, B = parse_combo(combo)
    total = len(W) + len(B)
    squares = list(range(64))
    picked = rng.sample(squares, total)
    w_sq = picked[:len(W)]
    b_sq = picked[len(W):]
    return w_sq, b_sq

def sample_random_position(
    combo: str,
    side_to_move: str = "w",
    *,
    seed: Optional[int] = None,
    max_tries: int = 100_000,
    require_nonterminal: bool = False,
    use_smart_bias: bool = True,
) -> Optional[str]:
    """
    Return a single random VALID FEN for `combo` (e.g., 'KRvK'), or None if not found.
      - `require_nonterminal=True` ensures the sampled position is not already game-over.
      - `use_smart_bias=True` biases kings (faster to find sensible positions).
    """
    rng = random.Random(seed)
    W, B = parse_combo(combo)
    wiK, biK = W.index("K"), B.index("k")

    for _ in range(max_tries):
        # choose placement strategy
        w_sq, b_sq = (_smart_squares_for(combo, rng) if use_smart_bias
                      else _uniform_squares_for(combo, rng))

        # quick illegal filters
        if _kings_adjacent(w_sq[wiK], b_sq[biK]):
            continue
        if any(_bad_pawn(p,s) for p,s in zip(W, w_sq)):  # no white pawn on rank 8
            continue
        if any(_bad_pawn(p,s) for p,s in zip(B, b_sq)):  # no black pawn on rank 1
            continue

        fen = _build_fen(list(zip(W, w_sq)), list(zip(B, b_sq)), side_to_move)

        # strict validity (if python-chess available)
        if not _fen_is_valid(fen):
            continue

        # load in your C++ engine as final sanity check
        g = cp.Game()
        try:
            g.reset_from_fen(fen)
        except Exception:
            continue

        if require_nonterminal and g.is_game_over():
            continue

        return fen

    return None

def sample_random_positions(
    combo: str,
    side_to_move: str = "w",
    *,
    count: int = 50,
    seed: Optional[int] = None,
    max_tries: int = 300_000,
    require_nonterminal: bool = False,
    use_smart_bias: bool = True,
) -> List[str]:
    """
    Collect up to `count` random valid FENs for `combo`.
    """
    rng = random.Random(seed)
    out, seen = [], set()
    tries = 0

    while len(out) < count and tries < max_tries:
        tries += 1
        fen = sample_random_position(
            combo,
            side_to_move=side_to_move,
            seed=rng.randrange(1<<30),
            max_tries=1_000,  # keep each inner attempt bounded
            require_nonterminal=require_nonterminal,
            use_smart_bias=use_smart_bias,
        )
        if fen and fen not in seen:
            seen.add(fen)
            out.append(fen)

    return out
