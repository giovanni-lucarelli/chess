# eval/evaluator.py
import time, pandas as pd

def evaluate(move_fn, fens, env_factory, dtm_oracle, optimal_moves=None, budget=None, max_plies=300):
    rows=[]
    for fen in fens:
        env = env_factory(fen)
        total_s, n = 0.0, 0
        first = None
        while env.steps() < max_plies and not env.is_terminal():
            t0 = time.perf_counter()
            uci = move_fn(env.to_fen(), budget=budget)  # greedy or MCTS
            total_s += (time.perf_counter() - t0); n += 1
            if first is None: first = uci
            sr = env.step(uci)
            if sr.done: break

        plies = env.steps()
        # mate = success
        try: success = int(env.state().is_checkmate())
        except: success = 0

        dtm = dtm_oracle[fen] if isinstance(dtm_oracle, dict) else int(dtm_oracle(fen))
        gap = (plies - dtm) if success else None
        if gap is not None and gap < 0: gap = 0

        top1 = None
        if optimal_moves and first:
            try: top1 = int(first in optimal_moves(fen))
            except: top1 = None

        rows.append({
            "fen": fen,
            "plies_policy": plies,
            "dtm_oracle": dtm,
            "gap": gap,
            "success": success,
            "ms_per_move": 1000.0 * total_s / max(n,1),
            "top1": top1,
            "budget": budget,
        })
    return pd.DataFrame(rows)
