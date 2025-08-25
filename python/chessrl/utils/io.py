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
    with open(path) as f:
        table = {row['fen']: row['uci'] for row in (json.loads(line) for line in f)}
    return table

def load_vf_parquet(path: str):
    return pd.read_parquet(path, engine='pyarrow').set_index("fen")["V"].to_dict()

def save_values(states: list[str], values, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"fen": states, "V": values}).to_parquet(path, index=False)
