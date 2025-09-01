import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from chessrl import Env, SyzygyDefender
from chessrl import chess_py as cp
from chessrl.utils.move_idx import build_move_mappings
from policy_mlp import PolicyMLP, extract_coords7_from_fen
from value_features import value_features_phi8, value_features_phi16, ValueMLP

from collections import deque
import torch.nn.functional as F

move_to_idx, idx_to_move = build_move_mappings()

def get_legal_move_indices(env):
    legal = []
    for move in env.state().legal_moves(cp.Color.WHITE):
        m = cp.Move.to_uci(move)[:4]
        if m in move_to_idx:
            legal.append(move_to_idx[m])
    return legal

class ActorCritic:
    def __init__(self, tb_path, gamma=0.99, lr_v=1e-3, lr_a=1e-3,
                 hidden=(128,128,64), defender=None):
        self.gamma = gamma
        self.defender = defender

        # actor
        self.policy = PolicyMLP(action_size=4096, hidden=hidden)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_a)

        # critic (neural)
        self.value = ValueMLP(in_dim=8, hidden=(64,64))
        self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_v)

        # (optional) extras you already had
        self.entropy_coeff = 0.01
        self.lam = 0.95

    # handy tensor maker
    def _to_t(self, phi, device):
        return torch.as_tensor(phi, dtype=torch.float32, device=device).view(1, -1)

    @torch.no_grad()
    def V(self, phi):
        device = next(self.value.parameters()).device
        return float(self.value(self._to_t(phi, device)).item())

    def update_step_td0(self, phi_s, logp, r, phi_sp, done, ent=None):
        """
        TD(0) target:  y = r + gamma * V(s')   (0 if terminal)
        advantage:     A = (y - V(s)) .detach()
        """
        device = next(self.policy.parameters()).device
        self.value.to(device)

        # tensors
        phi_s_t  = self._to_t(phi_s, device)                    # [1,16]
        v_s      = self.value(phi_s_t)                           # [1]
        with torch.no_grad():
            if done or (phi_sp is None):
                v_sp = torch.zeros_like(v_s)
            else:
                v_sp = self.value(self._to_t(phi_sp, device))

            target = torch.as_tensor(r, dtype=torch.float32, device=device) + self.gamma * v_sp
            adv    = (target - v_s).detach()

        # critic loss
        v_loss = F.mse_loss(v_s, target)

        # actor loss (+ entropy if available)
        if isinstance(ent, torch.Tensor) and ent.requires_grad:
            a_loss = -(adv * logp + self.entropy_coeff * ent)
        else:
            a_loss = -(adv * logp) if logp.requires_grad else None

        # step critic
        self.opt_v.zero_grad()
        v_loss.backward()
        clip_grad_norm_(self.value.parameters(), 1.0)
        self.opt_v.step()

        # step actor (skip if heuristic produced a constant logp)
        if a_loss is not None and a_loss.requires_grad:
            self.opt_pi.zero_grad()
            a_loss.backward()
            clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt_pi.step()

        return float((a_loss if a_loss is not None else adv).detach().item()), float(v_loss.detach().item())


    def td_error(self, r, phi_s, phi_sp, done):
        phi_s = self._as_phi1d(phi_s)
        v_s = float(self.w @ phi_s)
        v_sp = 0.0 if (done or phi_sp is None) else float(self.w @ self._as_phi1d(phi_sp))
        return r + self.gamma * v_sp - v_s

    def _gae_targets(self, rewards, values, dones, gamma=None, lam=None):
        gamma = self.gamma if gamma is None else gamma
        lam   = self.lam   if lam   is None else lam
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nonterm = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * values[t+1] * nonterm - values[t]
            last  = delta + gamma * lam * nonterm * last
            adv[t] = last
        targets = adv + values[:-1]
        return adv, targets

    def train(self, fens, epochs=3, max_steps=128, device=None):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(device).train()
        self.value.to(device).train()

        # keep only White-to-move
        fens = [f for f in fens if " w " in f]

        living_cost = 0.05          # NEGATIVE per white move
        all_losses, all_rewards = [], []

        for ep in range(epochs):
            np.random.shuffle(fens)
            pbar = tqdm(fens, desc=f"Epoch {ep+1}/{epochs}")
            for start_fen in pbar:
                s_fen = start_fen
                done  = False
                steps = 0
                ep_reward = 0.0

                # per-episode buffers
                phis, vals, logps, ents, rews, dns = [], [], [], [], [], []

                while not done and steps < max_steps:
                    env = Env.from_fen(
                        s_fen,
                        gamma=1.0,
                        defender=self.defender,
                        absorb_black_reply=True,
                        two_ply_cost=living_cost,   # ~ -0.05 each non-terminal step
                        draw_penalty=1.0,
                        checkmate_reward=+1.0
                    )
                    if " w " not in s_fen: break
                    legal = get_legal_move_indices(env)
                    if not legal: break

                    # features + current V(s)
                    phi_s = value_features_phi8(s_fen)
                    v_s   = self.value(self._to_t(phi_s, device))  # [1] tensor

                    phis.append(phi_s)                              # keep np arrays
                    vals.append(float(v_s.detach().cpu()))          # scalar

                    # policy step
                    a_idx, logp, ent = self.policy.get_action(env, legal)
                    uci   = idx_to_move[a_idx]
                    step  = env.step(uci)
                    s_next = env.state().to_fen()
                    r_env  = step.reward
                    done   = step.done

                    # (optional) add shaping here if you want:
                    # r = r_env + kappa * (self.gamma*Phi(s_next) - Phi(s_fen))
                    r = float(r_env)

                    # store for GAE
                    logps.append(logp)
                    ents.append(ent)
                    rews.append(r)
                    dns.append(bool(done))
                    ep_reward += r
                    all_rewards.append(r)

                    s_fen = s_next
                    steps += 1

                # bootstrap V(s_T)=0
                vals_np = np.array(vals + [0.0], dtype=np.float32)

                # advantages + lambda-returns
                adv, targets = self._gae_targets(rews, vals_np, dns)

                # ----- Critic: fit to λ-returns in one batch -----
                X = torch.as_tensor(np.vstack(phis), dtype=torch.float32, device=device)  # [T,16]
                y = torch.as_tensor(np.array(targets), dtype=torch.float32, device=device) # [T]
                v_pred = self.value(X)                                                     # [T]
                v_loss = torch.nn.functional.mse_loss(v_pred, y)

                self.v_optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.value.parameters(), 1.0)
                self.v_optimizer.step()

                # ----- Actor: PG with GAE (skip non-grad steps) -----
                loss_terms, ent_terms = [], []
                for t in range(len(rews)):
                    lp, en = logps[t], ents[t]
                    if isinstance(lp, torch.Tensor) and lp.requires_grad:
                        loss_terms.append(- torch.as_tensor(adv[t], dtype=torch.float32, device=device) * lp)
                        if isinstance(en, torch.Tensor) and en.requires_grad:
                            ent_terms.append(en)

                if loss_terms:
                    pi_loss = torch.stack(loss_terms).sum()
                    if ent_terms:
                        pi_loss = pi_loss - self.entropy_coeff * torch.stack(ent_terms).sum()
                    self.optimizer.zero_grad()
                    pi_loss.backward()
                    clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.optimizer.step()
                    all_losses.append(float(pi_loss.item()))
                else:
                    all_losses.append(0.0)

                pbar.set_postfix(steps=steps, ep_reward=f"{ep_reward:.2f}")

        return all_losses, all_rewards


    def save(self, policy_path="policy_mlp.pt", critic_path="critic_mlp.pt"):
        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.value.state_dict(), critic_path)

    def load(self, policy_path="policy_mlp.pt", critic_path="critic_mlp.pt", device=None):
        device = device or torch.device('cpu')
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        self.value.load_state_dict(torch.load(critic_path, map_location=device))
        self.policy.to(device).eval()
