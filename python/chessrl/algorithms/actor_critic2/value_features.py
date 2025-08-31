# value_features.py
import numpy as np
import torch
from chessrl.utils.fen_parsing import parse_fen

def value_features_phi8(fen: str) -> np.ndarray:
    """
    phi(s) = [1, d_edge(BK)/3, dxK/6, dyK/6, dxR/6, dyR/6, side, rook_present]
    If the rook is missing, set dxR=dyR=1.0 and rook_present=0.0.
    dx,dy use max(0, |Δ|-1) then scale.
    """
    b = parse_fen(fen)  # [8,8,12]

    def find_one(ch):
        idx = torch.nonzero(b[:, :, ch], as_tuple=False)
        return idx[0].tolist() if idx.numel() > 0 else None

    bk = find_one(11)  # black king
    wk = find_one(5)   # white king
    wr = find_one(3)   # white rook (may be None)
    assert bk is not None and wk is not None, f"Invalid KRK FEN (missing king): {fen}"

    bk_r, bk_c = bk
    wk_r, wk_c = wk

    def dxdy(wx, wy):
        dx = max(0, abs(wx - bk_c) - 1) / 6.0
        dy = max(0, abs(wy - bk_r) - 1) / 6.0
        return dx, dy

    dxk, dyk = dxdy(wk_c, wk_r)

    if wr is None:
        dxr = dyr = 1.0   # treat as "very far / useless" when rook is gone
        rook_present = 0.0
    else:
        wr_r, wr_c = wr
        dxr, dyr = dxdy(wr_c, wr_r)
        rook_present = 1.0

    d_edge = min(bk_r, 7 - bk_r, bk_c, 7 - bk_c) / 3.0
    side = 1.0 if fen.split()[1] == 'w' else -1.0

    return np.array([1.0, d_edge, dxk, dyk, dxr, dyr, side, rook_present], dtype=np.float32)
