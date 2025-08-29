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
        
        # Prepare batch of states for forward pass
        batch_states = []
        for fen in fens:
            state_tensor = parse_fen(fen).permute(2, 0, 1)  # [12, 8, 8]
            batch_states.append(state_tensor)
        
        batch_states = torch.stack(batch_states).to(self.device)  # [batch_size, 12, 8, 8]
        
        # Get policy outputs for all states at once
        logits = self.policy(batch_states)  # [batch_size, 4096]
        
        # Compute log probabilities for the actions taken
        log_probs = []
        for i, action_idx in enumerate(actions):
            action_logits = logits[i]
            action_probs = F.softmax(action_logits, dim=-1)
            log_prob = torch.log(action_probs[action_idx])
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs)
        deltas_torch = torch.tensor(deltas, dtype=torch.float32).to(self.device)
        
        # REINFORCE loss with baseline (advantage)
        loss = -(deltas_torch * log_probs).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
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
        
        # Perform batch update if buffer is full
        if len(self.experience_buffer['states']) >= self.batch_size:
            return self.batch_update()
        return None

    def train(self, endgames):
        losses = []
        rewards = []
        
        # Set policy to training mode
        self.policy.train()

        pbar = tqdm(enumerate(endgames), desc="Training Actor-Critic (Batch)", unit="episode", total=len(endgames))
        for endgame_idx, s in pbar:
            done = False
            x = self.obtain_features(s)
            counter = 0
            episode_rewards = []
            
            # Step by step (in the episode created by the endgame)
            while not done:
                env = Env.from_fen(
                    s,
                    defender = self.defender
                ) # create environment
                
                # Legal moves idx for this state
                legal_moves_idx = get_legal_move_indices(env)

                # Select action from policy (on device)
                a_idx, _ = self.policy.get_action(env, legal_moves_idx)  # We don't need log_prob here
                a = idx_to_move[a_idx] # returning UCI

                # Evolve one step
                step_result = env.step(a)

                new_s = env.state().to_fen()   
                r = step_result.reward  
                done = step_result.done
                if counter == config['max_steps']:
                    done = True   
                
                new_x = self.obtain_features(new_s)
                
                # Add experience to buffer and potentially update
                loss = self.add_experience(x, a_idx, r, new_x, done, s)  # Pass action index and current FEN
                
                if loss is not None:
                    losses.append(loss)
                
                episode_rewards.append(r)
                
                s = new_s # update state
                x = new_x # update features
                counter += 1
            
            # Store average reward for this episode
            if episode_rewards:
                rewards.append(np.mean(episode_rewards))

            # Save checkpoint 
            if (endgame_idx + 1) % 10000 == 0:
                self.save_checkpoint(f'output/checkpoint_ac_{endgame_idx + 1}.pth')

        # Final batch update for remaining experiences
        if self.experience_buffer['states']:
            final_loss = self.batch_update()
            if final_loss is not None:
                losses.append(final_loss)
        
        return losses, rewards
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {filepath}")