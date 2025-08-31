#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../../')

# utils
import numpy as np
import torch  
import torch.nn.functional as F
import logging
from tqdm import tqdm 
from chessrl.utils.load_config import load_config
import torch.nn as nn

import os
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# chess
from chessrl import Env, SyzygyDefender
from chessrl import chess_py as cp
from chessrl.algorithms.actor_critic.policy import Policy
from chessrl.utils.move_idx import build_move_mappings
from chessrl.utils.fen_parsing import parse_fen

move_to_idx, idx_to_move = build_move_mappings()

def get_legal_move_indices(env):
    """
    Get indices of legal moves for the current position.
    """
    legal_moves_idx = []
    
    for move in env.state().legal_moves(cp.Color.WHITE):
        move_str = cp.Move.to_uci(move)[:4]
        if move_str in move_to_idx:
            legal_moves_idx.append(move_to_idx[move_str])
    return legal_moves_idx

class ActorCritic():
    def __init__(self,  
                 tb_path: str = config['tb_path'],
                 gamma=1, 
                 lr_v=0.025,
                 lr_a=0.001,
                 batch_size=32):
        """
        Calculates optimal policy using in-policy Temporal Difference control
        Approximate V-value for states S!
        """        
        # Device setup - use MPS if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS device for acceleration")
        else:
            self.device = torch.device("cpu")
            logger.info("MPS not available, using CPU")
        
        # the discount factor
        self.gamma = gamma
        
        # the learning rate for value
        self.lr_v = lr_v
        
        # batch size for training
        self.batch_size = batch_size
        
        # Stores the Value Approximation weights (stays on CPU for tabular method)
        self.w = np.zeros(9604)
        self.mult = [7,7,7,7,4]

        self.defender = SyzygyDefender(tb_path=tb_path)

        # Move policy to device
        self.policy = Policy().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_a)
        
        # Buffer for batch training
        self.experience_buffer = {
            'states': [],
            'actions': [],  # Store actions instead of log_probs
            'rewards': [],
            'next_states': [],
            'dones': [],
            'fens': []  # Store FENs for recomputing
        }

    def obtain_features(self, fen):
        """
        Extract features from FEN:
        - Distance of black king from nearest side of the board.
        - Horizontal & vertical distances of each white piece from black king,
        always ordered deterministically: K, Q, R, B, N, P.
        """
        board = parse_fen(fen)  # [8, 8, 12]
        
        # Find black king position
        bk_pos = torch.nonzero(board[:, :, 11], as_tuple=False)
        if bk_pos.size(0) == 0:
            raise ValueError("No black king found in FEN")
        bk_row, bk_col = bk_pos[0].tolist()
        
        # Distance of black king from side
        dist_side = min(bk_row, 7 - bk_row, bk_col, 7 - bk_col)
        features = [dist_side]
        
        # Fixed order of white pieces
        white_piece_order = {
            "K": 5,
            "Q": 4,
            "R": 3,
            "B": 2,
            "N": 1,
            "P": 0
        }
        
        for piece_symbol in ["K", "Q", "R", "B", "N", "P"]:
            piece_idx = white_piece_order[piece_symbol]
            positions = torch.nonzero(board[:, :, piece_idx], as_tuple=False).tolist()
            # Sort by row, then col
            positions.sort()
            
            for r, c in positions:
                dx = max(0, abs(c - bk_col) - 1)
                dy = max(0, abs(r - bk_row) - 1)
                features.extend([dx, dy])
        
        return features
    
    def features_to_index(self, f):
        """
        Convert feature list to a unique index for value function lookup.
        """
        if len(f) < 5:
            raise ValueError(f"Feature vector has insufficient elements: {len(f)} < 5")
        return (((f[0]*self.mult[1] + f[1])*self.mult[2] + f[2])*self.mult[3] + f[3])*self.mult[4] + f[4]
    
    def batch_update(self):
        """
        Performs batch update for both critic and actor.
        """
        if len(self.experience_buffer['states']) == 0:
            return 0.0
        
        # Convert buffer to appropriate formats
        states = self.experience_buffer['states']
        actions = self.experience_buffer['actions']
        rewards = np.array(self.experience_buffer['rewards'])
        next_states = self.experience_buffer['next_states']
        dones = np.array(self.experience_buffer['dones'])
        fens = self.experience_buffer['fens']
        
        # ---------------------
        # CRITIC UPDATE (CPU) -
        # ---------------------
        deltas = np.zeros(len(states))
        
        for i in range(len(states)):
            idx_s = self.features_to_index(states[i])
            
            if dones[i]:
                delta = rewards[i] - self.w[idx_s]
            else:
                idx_new_s = self.features_to_index(next_states[i])
                delta = rewards[i] + self.gamma * self.w[idx_new_s] - self.w[idx_s]
            
            deltas[i] = delta
            # Update value function
            self.w[idx_s] += self.lr_v * delta
        
        # -------------------------
        # ACTOR UPDATE (MPS/GPU) --
        # -------------------------
        
        # Prepare batch of states for forward pass (minimize CPU-GPU transfers)
        batch_states = []
        for fen in fens:
            state_tensor = parse_fen(fen).permute(2, 0, 1)  # [12, 8, 8]
            batch_states.append(state_tensor)
        
        # Single tensor transfer to device
        batch_states = torch.stack(batch_states).to(self.device, non_blocking=True)  # [batch_size, 12, 8, 8]
        
        # Get policy outputs for all states at once
        logits = self.policy(batch_states)  # [batch_size, 4096]
        
        # Check for NaN in logits before computing loss
        if torch.isnan(logits).any():
            logger.warning("NaN detected in policy logits, reinitializing network")
            # Reinitialize the policy network
            def init_weights(m):
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            self.policy.apply(init_weights)
            return [], []  # Skip this update
        
        # Compute log probabilities for the actions taken (with legal move masking)
        log_probs = []
        for i, action_idx in enumerate(actions):
            # Get legal moves for this state
            env = Env.from_fen(fens[i], gamma=self.gamma, defender=self.defender)
            legal_moves_idx = get_legal_move_indices(env)
            
            if not legal_moves_idx or action_idx not in legal_moves_idx:
                # Skip this sample if no legal moves or action is illegal
                logger.warning(f"Illegal action {action_idx} or no legal moves for FEN: {fens[i]}")
                continue
            
            # Filter logits to legal moves only
            action_logits = logits[i]
            legal_logits = action_logits[legal_moves_idx]
            
            # Apply numerical stability for softmax on legal moves only
            legal_logits = torch.clamp(legal_logits, min=-50, max=50)
            legal_logits = legal_logits - torch.max(legal_logits)
            legal_probs = F.softmax(legal_logits, dim=-1)
            
            # Add epsilon to prevent log(0)
            legal_probs = legal_probs + 1e-8
            legal_probs = legal_probs / torch.sum(legal_probs)  # Renormalize
            
            # Find the action's position in the legal moves list
            action_pos = legal_moves_idx.index(action_idx)
            log_prob = torch.log(legal_probs[action_pos])
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs)
        deltas_torch = torch.tensor(deltas, dtype=torch.float32).to(self.device, non_blocking=True)
        
        # Check for NaN in loss computation
        if torch.isnan(log_probs).any():
            logger.warning("NaN detected in log probabilities, skipping update")
            return [], []
        
        # REINFORCE loss with baseline (advantage)
        loss = -(deltas_torch * log_probs).mean()
        
        # Check if loss is NaN
        if torch.isnan(loss):
            logger.warning("NaN detected in loss, skipping update")
            return [], []
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Check for NaN gradients
        has_nan_grad = False
        for param in self.policy.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            logger.warning("NaN gradients detected, skipping optimizer step")
            return [], []
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Clear buffer
        self.experience_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'fens': []
        }
        
        return loss.item()
    
    def add_experience(self, s, action, r, new_s, done, fen):
        """
        Add experience to buffer and trigger batch update if buffer is full.
        """
        self.experience_buffer['states'].append(s)
        self.experience_buffer['actions'].append(action)
        self.experience_buffer['rewards'].append(r)
        self.experience_buffer['next_states'].append(new_s)
        self.experience_buffer['dones'].append(done)
        self.experience_buffer['fens'].append(fen)
        
        # Perform batch update more frequently for better GPU utilization
        # Update when buffer reaches 75% capacity or has accumulated significant data
        buffer_size = len(self.experience_buffer['states'])
        update_threshold = max(self.batch_size // 4, 8)  # Update every 1/4 batch or min 8 samples
        
        if buffer_size >= update_threshold:
            return self.batch_update()
        return None

    def train(self, endgames):
        losses = []
        rewards = []
        
        # Set policy to training mode
        self.policy.train()
        
        # Convert endgames to list if it's a generator
        endgames_list = list(endgames)
        total_episodes = len(endgames_list)
        
        # Initialize parallel environments
        max_parallel = min(self.batch_size, 32)  # Limit to prevent memory issues
        episode_idx = 0
        
        pbar = tqdm(total=total_episodes, desc="Training Actor-Critic (True Batch)", unit="episode")
        
        while episode_idx < total_episodes:
            # Initialize batch of episodes
            active_episodes = []
            
            # Start new episodes up to batch size
            for _ in range(min(max_parallel, total_episodes - episode_idx)):
                if episode_idx >= total_episodes:
                    break
                    
                start_fen = endgames_list[episode_idx]
                env = Env.from_fen(start_fen, defender=self.defender)
                
                active_episodes.append({
                    'env': env,
                    'fen': start_fen,
                    'counter': 0,
                    'episode_rewards': [],
                    'episode_idx': episode_idx
                })
                
                episode_idx += 1
            
            # Run episodes in parallel until all complete
            while active_episodes:
                # Prepare batch data
                batch_envs = [ep['env'] for ep in active_episodes]
                legal_moves_list = [get_legal_move_indices(env) for env in batch_envs]
                
                # Batch inference for all active episodes
                actions, _ = self.policy.batch_get_actions(batch_envs, legal_moves_list)
                
                # Process each episode step
                completed_episodes = []
                for i, (episode, action_idx) in enumerate(zip(active_episodes, actions)):
                    env = episode['env']
                    
                    # Convert action index to move
                    try:
                        action = idx_to_move[action_idx]
                    except (KeyError, IndexError):
                        # Fallback to random legal move if action is invalid
                        legal_moves = list(env.legal_moves)
                        if legal_moves:
                            action = str(legal_moves[0])
                        else:
                            completed_episodes.append(i)
                            continue
                    
                    # Execute step
                    current_fen = env.state().to_fen()
                    current_x = self.obtain_features(current_fen)
                    
                    step_result = env.step(action)
                    
                    new_fen = env.state().to_fen()
                    new_x = self.obtain_features(new_fen)
                    r = step_result.reward
                    done = step_result.done
                    
                    episode['counter'] += 1
                    episode['episode_rewards'].append(r)
                    
                    # Check termination conditions
                    if done or episode['counter'] >= config['max_steps']:
                        done = True
                        completed_episodes.append(i)
                    
                    # Add experience to buffer
                    loss = self.add_experience(current_x, action_idx, r, new_x, done, current_fen)
                    if loss is not None:
                        losses.append(loss)
                
                # Remove completed episodes (in reverse order to maintain indices)
                for i in sorted(completed_episodes, reverse=True):
                    episode = active_episodes.pop(i)
                    if episode['episode_rewards']:
                        avg_reward = np.mean(episode['episode_rewards'])
                        rewards.append(avg_reward)
                    pbar.update(1)
                    
                    # Save checkpoint 
                    if (episode['episode_idx'] + 1) % 10000 == 0:
                        self.save_checkpoint(f'output/checkpoint_ac_{episode["episode_idx"] + 1}.pth')
                    
                    # Update progress bar
                    if losses:
                        pbar.set_postfix(
                            loss=f"{losses[-1]:.4f}",
                            avg_reward=f"{rewards[-1] if rewards else 0:.4f}",
                            active=len(active_episodes)
                        )
        
        # Process any remaining experiences in buffer
        if self.experience_buffer['states']:
            final_loss = self.batch_update()
            if final_loss is not None:
                losses.append(final_loss)
        
        pbar.close()
        return losses, rewards
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {filepath}")