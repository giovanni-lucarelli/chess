# value_features.py
import numpy as np
import torch
from chessrl.utils.fen_parsing import parse_fen

# value_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueMLP(nn.Module):
    def __init__(self, in_dim=8, hidden=(64,64)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2),     nn.Tanh(),
            nn.Linear(h2, 1)
        )

    def forward(self, x):              # x: [B, in_dim]
        return self.net(x).squeeze(-1) # -> [B]


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

def value_features_phi16(fen: str) -> np.ndarray:
    """
    16-D linear features for KRK value:
    [1,
     d_edgeBK, d_cornerBK,
     dInf(WK,BK), dMan(WK,BK),
     dInf(WR,BK), dMan(WR,BK),
     aligned, cut_w, opp,
     rook_adjBK,         # rook Chebyshev distance == 1
     same_file, same_rank,
     parity_sqBK,        # (-1)^(file+rank)
     side, rook_present
    ]
    All distances scaled to [0,1].
    """
    b = parse_fen(fen)  # [8,8,12]

    def find(ch):
        idx = torch.nonzero(b[:,:,ch], as_tuple=False)
        return idx[0].tolist() if idx.numel() else None

    bk = find(11); wk = find(5); wr = find(3)
    assert bk and wk, f"Invalid KRK FEN: {fen}"
    bk_r, bk_c = bk
    wk_r, wk_c = wk
    rook_present = 1.0 if wr else 0.0
    if wr:
        wr_r, wr_c = wr

    # helpers
    def d_inf(x1,y1,x2,y2): return max(abs(x1-x2), abs(y1-y2)) / 7.0
    def d_man(x1,y1,x2,y2): return (abs(x1-x2) + abs(y1-y2)) / 14.0

    # (a) BK distances to edge/corner
    d_edge = min(bk_r, 7-bk_r, bk_c, 7-bk_c) / 3.0
    d_corner = min(
        d_inf(bk_c,bk_r,0,0), d_inf(bk_c,bk_r,0,7),
        d_inf(bk_c,bk_r,7,0), d_inf(bk_c,bk_r,7,7)
    )

    # (b) distances to BK
    dinf_k = d_inf(wk_c,wk_r,bk_c,bk_r)
    dman_k = d_man(wk_c,wk_r,bk_c,bk_r)
    if wr:
        dinf_r = d_inf(wr_c,wr_r,bk_c,bk_r)
        dman_r = d_man(wr_c,wr_r,bk_c,bk_r)
    else:
        dinf_r = dman_r = 1.0

    # (c) alignment & cut width (how "boxed" BK is if aligned)
    same_file = 1.0 if (rook_present and wr_c == bk_c) else 0.0
    same_rank = 1.0 if (rook_present and wr_r == bk_r) else 0.0
    aligned = 1.0 if (same_file or same_rank) else 0.0
    cut_w = 0.0
    if rook_present:
        if same_file:   # vertical fence: how close BK is to vertical edge
            cut_w = min(bk_r, 7-bk_r) / 3.0
        elif same_rank: # horizontal fence
            cut_w = min(bk_c, 7-bk_c) / 3.0

    # (d) opposition (kings same file/rank with one square between)
    opp = 1.0 if ((wk_c == bk_c and abs(wk_r - bk_r) == 2) or
                  (wk_r == bk_r and abs(wk_c - bk_c) == 2)) else 0.0

    # (e) rook safety (adjacent to BK is risky)
    rook_adj = 1.0 if (rook_present and max(abs(wr_c-bk_c), abs(wr_r-bk_r)) == 1) else 0.0

    # (f) parity (color of BK square) and side to move
    parity_sqBK = 1.0 if ((bk_r + bk_c) & 1) == 0 else -1.0
    side = 1.0 if fen.split()[1] == 'w' else -1.0

    return np.array([
        1.0,
        d_edge, d_corner,
        dinf_k, dman_k,
        dinf_r, dman_r,
        aligned, cut_w, opp,
        rook_adj,
        same_file, same_rank,
        parity_sqBK,
        side, rook_present
    ], dtype=np.float32)