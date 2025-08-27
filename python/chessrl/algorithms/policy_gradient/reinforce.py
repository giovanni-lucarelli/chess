#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../../')

# utils
import numpy as np
import torch 
from torch import nn 
import logging
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from chessrl.utils.load_config import load_config
from chessrl.utils.fen_parsing import parse_fen
from typing import List
from chessrl.utils.io import save_policy_jsonl
import random

import os
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# chess
from chessrl import Env, SyzygyDefender
from chessrl import chess_py as cp
from chessrl.algorithms.policy_gradient.policy import Policy
from chessrl.utils.move_idx import build_move_mappings

move_to_idx, idx_to_move = build_move_mappings()

def get_legal_move_indices(env):
    """
    Get indices of legal moves for the current position.
    """
    legal_moves_idx = []
    
    # Get legal moves for the current side
    for move in env.state().legal_moves(cp.Color.WHITE):
        move_str = cp.Move.to_uci(move)[:4]
        if move_str in move_to_idx:
            legal_moves_idx.append(move_to_idx[move_str])
    return legal_moves_idx

class REINFORCE:
    def __init__(
            self,
            lr: float = config['alpha'],
            gamma: float = config['gamma'],
            max_steps: int = config['max_steps'],
            tb_path: str = '../../../../tablebase/krk/'
    ):
        self.lr = lr 
        self.gamma = gamma 
        self.policy = Policy(
            input_channels=12,
            filters=128,
            residual_blocks=6,
            policy_head_filters=16,
            action_size=4096
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.dtm_history = []
        self.max_steps = max_steps
        
        # Initialize Syzygy defender for black moves
        self.defender = SyzygyDefender(tb_path=tb_path)
    
    def sample_episode(self, env, max_steps: int = config['max_steps']):
        """
        Sample a trajectory using the current policy for White
        and oracle (best possible move) for black.
        """
        
        # Store episode data
        states = []      # FEN strings
        actions = []     # Move indices
        log_probs = []   # Log probabilities of actions
        rewards = []     # Immediate rewards
        
        for step in range(max_steps):
            current_fen = env.to_fen()

            legal_moves = get_legal_move_indices(env)

            if not legal_moves:
                break  # No legal moves available
            
            # Get action probabilities
            fen_tensor = parse_fen(current_fen).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                action_probs = self.policy.get_action_probs(fen_tensor, legal_moves)
                action_probs = action_probs.squeeze(0).cpu().numpy()
                log_action_probs = self.policy.get_action_probs(fen_tensor, legal_moves, log=True)
                log_action_probs_tensor = log_action_probs.squeeze(0).cpu()
            
            # Extract probabilities for legal moves only
            legal_probs = action_probs[legal_moves]  # Remove the extra [0,] indexing
            # Add small epsilon to avoid zero probabilities
            legal_probs = legal_probs + 1e-8
            # Normalize to ensure they sum to 1
            legal_probs = legal_probs / legal_probs.sum()
            
            # Sample from legal moves (alternative to epsilon-greedy policy)
            selected_legal_idx = np.random.choice(len(legal_moves), p=legal_probs)
            action_idx = legal_moves[selected_legal_idx]
            
            # Convert to move
            move_str = idx_to_move[action_idx]
            move = cp.Move.from_strings(env.state(), move_str[:2], move_str[2:4])
            
            # Take step
            step_result = env.step(move)

            # Store data
            states.append(current_fen)
            actions.append(move_str)
            log_probs.append(log_action_probs_tensor[action_idx])
            rewards.append(step_result.reward)  
            
            if env.state().is_game_over():
                break
        
        DTM = float('inf')
        
        if env.state().is_checkmate():
            DTM = 2*(len(states) - 0.5)
            # apply a step penalty only if checkmate (-1 for every move) to encourage faster checkmates
            rewards[-1] = rewards[-1] - len(states)

        return states, actions, log_probs, rewards, DTM

    def calculate_returns(self, rewards: List[float]) -> List[float]:
        """
        Calculate discounted returns for each timestep.
        """
        returns = []
        G = 0
        
        # Calculate returns backwards
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
            
        return returns

    def train_episode(self, env):
        """
        Train on a single episode (legacy method for backward compatibility).
        """
        states, actions, _, rewards, DTM = self.sample_episode(env)
        
        # Calculate returns for each time step
        returns = self.calculate_returns(rewards)
        
        # Recompute log probabilities with gradients enabled
        log_probs_list = []
        for i, (state, action) in enumerate(zip(states, actions)):
            # Parse FEN and get legal moves for this specific state
            fen_tensor = parse_fen(state).permute(2, 0, 1).unsqueeze(0)
            
            # Create temporary env to get legal moves for this state
            temp_env = Env.from_fen(state, defender=self.defender)
            legal_moves = get_legal_move_indices(temp_env)
            
            # Get action index from action string
            action_idx = move_to_idx[action]
            
            # Get log probability with gradients
            log_action_probs = self.policy.get_action_probs(fen_tensor, legal_moves, log=True)
            log_prob = log_action_probs[0, action_idx]
            log_probs_list.append(log_prob)
        
        # Calculate loss with numerical stability
        log_probs_tensor = torch.stack(log_probs_list)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # REINFORCE loss: -E[log π(a|s) * G]
        loss = -torch.mean(log_probs_tensor * returns_tensor)

        # Check for NaN or inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss detected: {loss.item()}, skipping update")
            return 0.0, DTM
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), DTM

    def train(self, 
              endgames):
        """
        Train the policy on multiple endgame positions with batch updates.
        
        Args:
            endgames: List of endgame FENs to train on
        """

        accumulated_dtm = []
        accumulated_loss = []
        wins = 0
        total_episodes = 0

        # Create progress bar
        with tqdm(total=len(endgames), desc="Training") as pbar:
            for i, endgame in enumerate(endgames):
                env = Env.from_fen(
                    endgame,
                    defender=self.defender, 
                    two_ply_cost=0.0,
                    draw_penalty=50.0,
                    checkmate_reward=100.0  
                )
                loss, DTM = self.train_episode(env)

                if loss != 0.0:
                    accumulated_loss.append(loss)

                # Store DTM for plotting
                if DTM is not None and DTM != float('inf'):
                    accumulated_dtm.append(DTM)
                    wins += 1
                
                # Update progress bar with recent metrics
                mean_loss = np.mean(accumulated_loss)
                pbar.set_postfix({
                    'Loss': f'{mean_loss:.3f}' if mean_loss is not None else 'N/A',
                    'Wins': wins
                })
                pbar.update(1)

        self.save_model()

    def save_model(self, filepath=config['filepath_train']):
        """
        Save the trained policy weights to a file.
        """
        logger.info(f'Saving model with filepath {filepath}')
        torch.save(self.policy.state_dict(), filepath)
    
    def load_model(self, filepath=config['filepath_train']):
        """
        Load trained policy weights from a file.
        """
        logger.info(f'Loading model with filepath {filepath}')
        try:
            self.policy.load_state_dict(torch.load(filepath))
            self.policy.eval()
        except FileNotFoundError:
            logger.error(f"File {filepath} not found. Using randomly initialized weights.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
