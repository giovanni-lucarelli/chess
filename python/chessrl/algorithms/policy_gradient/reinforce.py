import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from chessrl import Env, SyzygyDefender
from chessrl import chess_py as cp
from chessrl.utils.move_idx import build_move_mappings
from chessrl.algorithms.policy_gradient.policy import Policy

move_to_idx, idx_to_move = build_move_mappings()

def get_legal_move_indices(env):
    legal = []
    for move in env.state().legal_moves(cp.Color.WHITE):
        m = cp.Move.to_uci(move)[:4]
        if m in move_to_idx:
            legal.append(move_to_idx[m])
    return legal

class Reinforce:
    def __init__(self, 
                 tb_path, 
                 gamma=0.99, 
                 lr=1e-2,
                 hidden=(128,128,64)
                 ):
        self.gamma = gamma
        self.defender = SyzygyDefender(tb_path=tb_path)

        self.policy = Policy(action_size=4096, hidden=hidden)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def sample_episode(self, fen, max_steps=50):
        episode_data = {
            'actions': [],
            'rewards': [],
        }
        
        done=False
        steps=0
        while not done and steps < max_steps:
            env = Env.from_fen(
                fen=fen,
                gamma=self.gamma,
                defender=self.defender,
                absorb_black_reply=True,
                two_ply_cost=1.0,
                draw_penalty=1.0,
                checkmate_reward=1.0
            )
            legal_moves_idx = get_legal_move_indices(env)
            if not legal_moves_idx:
                break

            action_idx, logp = self.policy.get_action(env, legal_moves_idx)
            action_uci = idx_to_move[action_idx]
            episode_data['actions'].append((action_uci, logp))
            step = env.step(action_uci)
            episode_data['rewards'].append(step.reward)

            fen = env.to_fen()
            done = step.done
            steps += 1

        return episode_data 
    
    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def compute_loss(self, actions, returns):
        device = next(self.policy.parameters()).device
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        log_probs = torch.stack([logp for _, logp in actions]).to(device)
        loss = -torch.sum(log_probs * returns)
        return loss

    def update_policy(self, episode_data):
        actions = episode_data['actions']
        rewards = episode_data['rewards']

        # Compute returns
        returns = self.compute_returns(rewards)

        # Update policy
        self.optimizer.zero_grad()
        loss = self.compute_loss(actions, returns)
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss

    def train(self, fens, epochs=3, max_steps=50, device=None):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(device).train()

        all_losses, all_rewards = [], []

        for epoch in range(epochs):
            np.random.shuffle(fens)
            pbar = tqdm(fens, desc=f"Epoch {epoch+1}/{epochs}")
            for start_fen in pbar:
                episode_data = self.sample_episode(fen=start_fen, max_steps=max_steps)
                loss = self.update_policy(episode_data)
                all_losses.append(loss)
                all_rewards.append(np.sum(episode_data['rewards']))

        return all_losses, all_rewards

    def save(self, policy_path="output/policy.pt"):
        torch.save(self.policy.state_dict(), policy_path)

    def load(self, policy_path="output/policy.pt", device=None):
        device = device or torch.device('cpu')
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        self.policy.to(device).eval()
