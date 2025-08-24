import time, pandas as pd
from chessrl.env import Env, SyzygyDefender

def evaluate(move_fn, fens, tb_path, dtm_oracle, optimal_moves=None, budget=None, max_plies=300):
    rows = []
    for fen in fens:
        env = Env.from_fen(
            fen, gamma=1.0, step_penalty=0.0,
            defender=SyzygyDefender(tb_path), absorb_black_reply=True
        )
        total_s, n = 0.0, 0
        first = None

        opt_hits = 0       # how many of our white moves matched the oracle
        opt_total = 0      # how many times the oracle gave us a set to compare against

        while env.steps() < max_plies and not env.is_terminal():
            # use the env's full, normalized FEN for both move_fn and oracle
            root_fen = env.to_fen()

            t0 = time.perf_counter()
            uci = move_fn(root_fen, budget=budget)  # greedy or MCTS
            total_s += (time.perf_counter() - t0); n += 1
            if first is None:
                first = uci

            # per-decision top-1: compare our move to oracle's optimal set at this state
            if optimal_moves and uci:
                try:
                    opts = optimal_moves(root_fen)   # set[str]
                    if opts:                         # only count if oracle gave us something
                        opt_total += 1
                        if uci in opts:
                            opt_hits += 1
                except Exception:
                    # if oracle fails, just skip this decision
                    pass

            sr = env.step(uci)
            if sr.done:
                break

        plies = env.steps()
        # success = checkmate for White
        try:
            success = int(env.state().is_checkmate())
        except Exception:
            success = 0

        # DTM oracle
        dtm = dtm_oracle[fen] if isinstance(dtm_oracle, dict) else int(dtm_oracle(fen))
        gap = (plies - dtm) if success else None
        if gap is not None and gap < 0:
            gap = 0

        # top1 over the whole game: fraction of white decisions that were oracle-optimal
        top1 = (opt_hits / opt_total) if opt_total > 0 else None

        rows.append({
            "fen": fen,
            "dtm_policy": plies,
            "dtm_oracle": dtm,
            "gap": gap,
            "success": success,
            "ms_per_move": 1000.0 * total_s / max(n, 1),
            "top1": top1,                 # per-episode fraction in [0,1]
            "top1_decisions": opt_total,  # how many decisions were evaluated
            "budget": budget,
        })

    return pd.DataFrame(rows)
