import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from chessrl import Env, SyzygyDefender
from chessrl import chess_py as cp
from chessrl.utils.move_idx import build_move_mappings
from policy_mlp import PolicyMLP
from value_features import value_features_phi8

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
                 gamma=0.99, 
                 lr_v=0.02, 
                 lr_a=1e-2,
                 hidden=(128,128,64)
                 ):
        self.gamma = gamma
        self.lr_v  = lr_v
        # critic weights for phi in R^8
        self.w = np.zeros(8, dtype=np.float32)
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

    def train(self, fens, epochs=3, max_steps=50, device=None):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(device).train()

        all_losses, all_rewards = [], []
        for ep in range(epochs):
            np.random.shuffle(fens)
            pbar = tqdm(fens, desc=f"Epoch {ep+1}/{epochs}")
            for start_fen in pbar:
                s_fen = start_fen
                done = False
                steps = 0
                phi_s = value_features_phi8(s_fen)
                ep_reward = 0.0

                while not done and steps < max_steps:
                    env = Env.from_fen(
                        s_fen,
                        gamma=1.0,
                        defender=self.defender,
                        absorb_black_reply=True,
                        two_ply_cost=1.0,   # -> reward -1 on non-terminal
                        draw_penalty=1.0,   # -> reward -1 on draw
                        checkmate_reward=1.0  # -> reward +1 on mate
                    )
                    legal = get_legal_move_indices(env)
                    if not legal:  # no legal white moves (shouldn’t happen for wtm)
                        break

                    a_idx, logp = self.policy.get_action(env, legal)
                    uci = idx_to_move[a_idx]
                    step = env.step(uci)

                    s_next = env.state().to_fen()
                    r = step.reward
                    done = step.done

                    if not done:
                        phi_sp = value_features_phi8(s_next)
                    else:
                        phi_sp = None

                    loss, _ = self.update_step(phi_s, logp, r, phi_sp, done)
                    all_losses.append(loss)
                    all_rewards.append(r)
                    ep_reward += r

                    s_fen, phi_s = s_next, phi_sp
                    steps += 1

                pbar.set_postfix(steps=steps, ep_reward=f"{ep_reward:.2f}")
        return all_losses, all_rewards

    def train_m1(self, fens, epochs=3, device=None):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(device).train()
        self.gamma = 0.0   # no bootstrap
        losses, sr_hist = [], []

        for ep in range(epochs):
            np.random.shuffle(fens)
            wins = 0; total = 0
            pbar = tqdm(fens, desc=f"M1 RL ep {ep+1}/{epochs}")
            for fen in pbar:
                # Just to get legal moves for White
                env = Env.from_fen(
                    fen,
                    two_ply_cost=0.0,           # irrelevant; we won’t roll out
                    draw_penalty=0.0,
                    checkmate_reward=0.0,
                    defender=self.defender
                )
                legal = get_legal_move_indices(env)
                if not legal:
                    continue

                # Sample action and compute single-step reward
                a_idx, logp = self.policy.get_action(env, legal)
                uci = idx_to_move[a_idx]

                env2 = Env.from_fen(
                    fen,
                    two_ply_cost=0.0,
                    draw_penalty=0.0,
                    checkmate_reward=0.0,
                    defender=self.defender
                )
                _ = env2.step(uci)
                is_mate = env2.state().is_checkmate()

                r = 1.0 if is_mate else -1.0
                phi_s = value_features_phi8(fen)

                # Done immediately; no next-state baseline
                loss, _ = self.update_step(phi_s, logp, r, phi_sp=None, done=True)
                losses.append(loss)

                wins += int(is_mate); total += 1
                if total % 200 == 0:
                    pbar.set_postfix(loss=f"{np.mean(losses[-200:]):.3f}", SR=f"{wins/total:.3f}")
            sr_hist.append(wins / max(1, total))
        return losses, sr_hist


    def save(self, policy_path="policy_mlp.pt", critic_path="critic_w.npy"):
        torch.save(self.policy.state_dict(), policy_path)
        np.save(critic_path, self.w)

    def load(self, policy_path="policy_mlp.pt", critic_path="critic_w.npy", device=None):
        device = device or torch.device('cpu')
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        self.w = np.load(critic_path).astype(np.float32)
        self.policy.to(device).eval()
