import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from chessrl import Env, SyzygyDefender
from chessrl import chess_py as cp
from chessrl.utils.move_idx import build_move_mappings
from .policy_mlp import PolicyMLP
from .value_features import value_features_phi16
import random

move_to_idx, idx_to_move = build_move_mappings()

def get_legal_move_indices(env):
    legal = []
    for move in env.state().legal_moves(cp.Color.WHITE):
        m = cp.Move.to_uci(move)[:4]
        if m in move_to_idx:
            legal.append(move_to_idx[m])
    return legal

class ActorCritic:
    def __init__(self, 
                 tb_path, 
                 gamma=0.9, 
                 lr_v=0.02, 
                 lr_a=1e-3,
                 hidden=(128,128,64)
                 ):
        self.gamma = gamma
        self.lr_v  = lr_v
        # critic weights for phi in R^8
        self.w = np.zeros(16, dtype=np.float32)
        self.defender = SyzygyDefender(tb_path=tb_path)

        self.policy = PolicyMLP(action_size=4096, hidden=hidden)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_a)

    def V(self, phi): return float(np.dot(self.w, phi))

    def td_error(self, r, phi_s, phi_sp, done):
        v_s = self.V(phi_s)
        v_sp = 0.0 if (done or phi_sp is None) else self.V(phi_sp)
        return r + self.gamma * v_sp - v_s

    def update_step(self, phi_s, logp, r, phi_sp, done):
        # critic: semi-gradient TD(0)
        delta = self.td_error(r, phi_s, phi_sp, done)
        self.w += self.lr_v * delta * phi_s

        # actor
        device = next(self.policy.parameters()).device
        delta_t = torch.tensor(delta, dtype=torch.float32, device=device)
        loss = -(delta_t * logp)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item()), float(delta)
    
    def train(self, fens, epochs=3, max_steps=128, device=None):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(device).train()

        all_losses, all_rewards = [], []
        all_ep_discounted = []

        for ep in range(epochs):
            np.random.shuffle(fens)
            pbar = tqdm(fens, desc=f"Epoch {ep+1}/{epochs}")
            for start_fen in pbar:
                s_fen = start_fen
                done = False
                steps = 0
                phi_s = value_features_phi16(s_fen)

                ep_reward = 0.0
                ep_return_discounted = 0.0
                pow_gamma = 1.0

                while not done and steps < max_steps:
                    env = Env.from_fen(
                        s_fen,
                        gamma=self.gamma,
                        defender=self.defender,
                        absorb_black_reply=True,
                        two_ply_cost=0.0,
                        draw_penalty=1.0,
                        checkmate_reward=1.0
                    )
                    legal = get_legal_move_indices(env)
                    if not legal:
                        break

                    a_idx, logp = self.policy.get_action(env, legal)
                    uci = idx_to_move[a_idx]
                    step = env.step(uci)

                    s_next = env.state().to_fen()
                    r = step.reward
                    done = step.done

                    # discounted return logging
                    ep_reward += r
                    ep_return_discounted += pow_gamma * r
                    pow_gamma *= self.gamma

                    if not done:
                        phi_sp = value_features_phi16(s_next)
                    else:
                        phi_sp = None

                    loss, _ = self.update_step(phi_s, logp, r, phi_sp, done)
                    all_losses.append(loss)
                    all_rewards.append(r)

                    s_fen, phi_s = s_next, phi_sp
                    steps += 1

                all_ep_discounted.append(ep_return_discounted)

                pbar.set_postfix(
                    steps=steps,
                    ep_reward=f"{ep_reward:.2f}",
                    G_discounted=f"{ep_return_discounted:.3f}"
                )

        return all_losses, all_rewards

    def save(self, policy_path="policy_mlp.pt", critic_path="critic_w.npy"):
        torch.save(self.policy.state_dict(), policy_path)
        np.save(critic_path, self.w)

    def load(self, policy_path="policy_mlp.pt", critic_path="critic_w.npy", device=None):
        device = device or torch.device('cpu')
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        self.w = np.load(critic_path).astype(np.float32)
        self.policy.to(device).eval()