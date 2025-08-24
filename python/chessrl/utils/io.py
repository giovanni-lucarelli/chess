import json, os, pandas as pd
from chessrl.chess_py import Move as CMove

def _to_uci(m):
    if m is None: return None
    try: return CMove.to_uci(m)
    except Exception: return str(m)

def save_policy_jsonl(policy: dict[str, object], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for fen, move in policy.items():
            f.write(json.dumps({"fen": fen, "uci": _to_uci(move)}) + "\n")

def load_policy_jsonl(path: str):
    table={}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            table[row["fen"]] = row["uci"]
    return lambda fen, budget=None: table.get(fen)

def save_values(states: list[str], values, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"fen": states, "V": values}).to_parquet(path, index=False)
