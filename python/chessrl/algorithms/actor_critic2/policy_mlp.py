# policy_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from chessrl.utils.fen_parsing import parse_fen

def _norm01_to_pm1(v):  # 0..7 -> [-1,1]
    return (v - 3.5) / 3.5

def extract_coords7_from_fen(fen: str) -> torch.Tensor:
    """
    Return [wx, wy, rx, ry, bx, by, side] normalized to [-1,1].
    Assumes parse_fen -> [8,8,12] with channels:
      white: P=0,N=1,B=2,R=3,Q=4,K=5 ; black king k=11
    """
    board = parse_fen(fen)  # [8,8,12]
    wk = torch.nonzero(board[:,:,5],  as_tuple=False)[0].tolist()  # row, col
    wr = torch.nonzero(board[:,:,3],  as_tuple=False)[0].tolist()
    bk = torch.nonzero(board[:,:,11], as_tuple=False)[0].tolist()
    wy, wx = wk; ry, rx = wr; by, bx = bk
    side = 1.0 if fen.split()[1] == 'w' else -1.0
    return torch.tensor([
        _norm01_to_pm1(wx), _norm01_to_pm1(wy),
        _norm01_to_pm1(rx), _norm01_to_pm1(ry),
        _norm01_to_pm1(bx), _norm01_to_pm1(by),
        side
    ], dtype=torch.float32)

class PolicyMLP(nn.Module):
    """
    Coordinate MLP: input 7 floats -> 4096 logits (masked to legal moves at sampling).
    """
    def __init__(self, action_size=4096, hidden=(128,128,64)):
        super().__init__()
        h1,h2,h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(7, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, action_size)
        )

    def forward(self, x7):  # x7: [B,7]
        return self.net(x7)

    @torch.no_grad()
    def predict_from_fen(self, fen: str):
        self.eval()
        device = next(self.parameters()).device
        x7 = extract_coords7_from_fen(fen).unsqueeze(0).to(device)
        probs = F.softmax(self.forward(x7), dim=-1)
        return int(torch.argmax(probs, dim=-1).item())

    # def get_action(self, env, legal_moves_idx):
    #     device = next(self.parameters()).device
    #     x7 = extract_coords7_from_fen(env.state().to_fen()).unsqueeze(0).to(device)
    #     logits = self.forward(x7)[0]
    #     legal_logits = logits[0, legal_moves_idx]        
    #     dist = Categorical(logits=legal_logits)
    #     a_off = dist.sample()
    #     logp = dist.log_prob(a_off)
    #     a_idx = legal_moves_idx[a_off.item()]
    #     return a_idx, logp

    
    # @torch.no_grad()
    # def get_action_greedy(self, env, legal_moves_idx):
    #     device = next(self.parameters()).device
    #     x7 = extract_coords7_from_fen(env.to_fen()).unsqueeze(0).to(device)
    #     logits = self.forward(x7)[0]
    #     j = torch.argmax(logits[legal_moves_idx]).item()
    #     return legal_moves_idx[j]

    def get_action(self, env, legal_moves_idx):
        device = next(self.parameters()).device
        x7 = extract_coords7_from_fen(env.state().to_fen()).unsqueeze(0).to(device)
        logits = self.forward(x7)  # [1, A]
        legal_idx = torch.as_tensor(legal_moves_idx, device=logits.device, dtype=torch.long)
        legal_logits = logits[0, legal_idx]  # [K]
        dist = Categorical(logits=legal_logits)
        a_off = dist.sample()
        logp = dist.log_prob(a_off)
        a_idx = legal_moves_idx[a_off.item()]
        return a_idx, logp

    @torch.no_grad()
    def get_action_greedy(self, env, legal_moves_idx):
        device = next(self.parameters()).device
        x7 = extract_coords7_from_fen(env.state().to_fen()).unsqueeze(0).to(device)
        logits = self.forward(x7)  # [1, A]
        legal_idx = torch.as_tensor(legal_moves_idx, device=logits.device, dtype=torch.long)
        j = torch.argmax(logits[0, legal_idx]).item()
        return legal_moves_idx[j]

