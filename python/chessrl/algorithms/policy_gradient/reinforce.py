#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../../')

# utils
import numpy as np
import torch  
import logging
from tqdm import tqdm 
from chessrl.utils.load_config import load_config
from chessrl.utils.fen_parsing import parse_fen_cached
from typing import List, Tuple, Dict


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

def get_device():
    """
    Get the best available device (MPS for Mac, CUDA for NVIDIA, CPU otherwise).
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

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

class REINFORCE:
    def __init__(
            self,
            lr: float = config['alpha'],
            gamma: float = config['gamma'],
            batch_size: int = 32,  
            tb_path: str = '../../../../tablebase/krk/'
    ):
        self.lr = lr 
        self.gamma = gamma 
        self.batch_size = batch_size
        
        # Setup device
        self.device = get_device()
        
        # Initialize policy on the correct device
        self.policy = Policy().to(self.device) 
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.dtm_history = []
        
        # Initialize Syzygy defender for black moves
        self.defender = SyzygyDefender(tb_path=tb_path)
        
        # Cache for parsed FENs to avoid recomputation
        self.fen_cache = {}
    
    def sample_episodes(self, envs: List[Env], max_steps: int = config['max_steps']) -> Tuple:
        """
        Sample multiple episodes in parallel using batch processing.
        Returns collected data for all episodes.
        """
        batch_size = len(envs)
        episodes_data = [{'states': [], 'actions': [], 'rewards': [], 'action_indices': [], 'done': False} 
                        for _ in range(batch_size)]
        
        for step in range(config['max_steps']):
            # Collect states and legal moves for active episodes
            batch_fens = []
            batch_legal_moves = []
            active_indices = []
            
            for i, env in enumerate(envs):
                if not episodes_data[i]['done']:
                    current_fen = env.to_fen()
                    legal_moves = get_legal_move_indices(env) # idx of actions
                    
                    if not legal_moves or env.state().is_game_over():
                        episodes_data[i]['done'] = True
                        continue
                    
                    batch_fens.append(current_fen)
                    batch_legal_moves.append(legal_moves)
                    active_indices.append(i)
            
            if not active_indices:
                break  # All episodes are done
            
            # Batch process active episodes
            if batch_fens:
                # Parse all FENs and stack into batch tensor
                fen_tensors = torch.stack([
                    parse_fen_cached(fen, self.fen_cache) for fen in batch_fens
                ]).to(self.device)
                
                # Get action probabilities for entire batch
                with torch.no_grad():
                    # Forward pass for entire batch
                    batch_logits = self.policy.forward(fen_tensors)
                    
                    # Process each episode in the batch
                    for batch_idx, env_idx in enumerate(active_indices):
                        legal_moves = batch_legal_moves[batch_idx]
                        logits = batch_logits[batch_idx]
                        
                        # Create mask for legal moves
                        mask = torch.zeros(4096, device=self.device)
                        mask[legal_moves] = 1
                        
                        # Apply mask and get probabilities
                        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
                        action_probs = torch.softmax(masked_logits, dim=-1)
                        log_probs = torch.log_softmax(masked_logits, dim=-1)
                        
                        # Sample action
                        legal_probs = action_probs[legal_moves].cpu().numpy()
                        legal_probs = legal_probs + 1e-8 # for numerical stability
                        legal_probs = legal_probs / legal_probs.sum()
                        
                        selected_legal_idx = np.random.choice(len(legal_moves), p=legal_probs) # alternative to epsilon-greedy method used in other algorithms
                        action_idx = legal_moves[selected_legal_idx]
                        
                        # Convert to move and take step
                        move_str = idx_to_move[action_idx]
                        move = cp.Move.from_strings(envs[env_idx].state(), 
                                                   move_str[:2], move_str[2:4])
                        step_result = envs[env_idx].step(move)
                        
                        # Store episode data
                        episodes_data[env_idx]['states'].append(batch_fens[batch_idx])
                        episodes_data[env_idx]['actions'].append(move_str)
                        episodes_data[env_idx]['rewards'].append(step_result.reward)
                        episodes_data[env_idx]['action_indices'].append(action_idx)
                        
                        if envs[env_idx].state().is_game_over():
                            episodes_data[env_idx]['done'] = True
                            if envs[env_idx].state().is_checkmate():
                                dtm = 2 * (len(episodes_data[env_idx]['states']) - 0.5)
                                self.dtm_history.append(dtm) # Store DTM for later analysis
        
        return episodes_data
    
    def calculate_returns(self, episodes_rewards: List[List[float]]) -> List[torch.Tensor]:
        """
        Calculate discounted returns for multiple episodes.
        """
        all_returns = []
        
        for rewards in episodes_rewards:
            returns = []
            G = 0
            for reward in reversed(rewards):
                G = reward + self.gamma * G
                returns.insert(0, G)
            all_returns.append(torch.tensor(returns, dtype=torch.float32, device=self.device))
        
        return all_returns
    
    def train_batch(self, episodes_data: List[Dict]) -> Tuple[float, List[float]]:
        """
        Train on a batch of episodes simultaneously.
        """
        # Calculate returns for all episodes
        all_rewards = [ep['rewards'] for ep in episodes_data]
        all_returns = self.calculate_returns(all_rewards)
        
        # Prepare batch tensors
        batch_states = []
        batch_actions = []
        batch_returns = []
        batch_legal_moves = []
        
        for ep_idx, episode in enumerate(episodes_data): # iterate through each episode in the batch
            for t in range(len(episode['states'])): # t indicates the step in the episode
                batch_states.append(episode['states'][t])
                batch_actions.append(episode['actions'][t])
                batch_returns.append(all_returns[ep_idx][t])
                
                # Get legal moves for this state
                temp_env = Env.from_fen(episode['states'][t], defender=self.defender)
                legal_moves = get_legal_move_indices(temp_env) # returns moves' idxs
                batch_legal_moves.append(legal_moves)
        
        if not batch_states:
            return 0.0, []
        
        # Convert to tensors
        state_tensors = torch.stack([
            parse_fen_cached(fen, self.fen_cache) for fen in batch_states
        ]).to(self.device) # shapte [N, 12, 8, 8] where N is total steps in batch
        
        returns_tensor = torch.stack(batch_returns) # shape [N]
        
        # Normalize returns for stability 
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Forward pass with gradients
        logits = self.policy.forward(state_tensors)
        
        # Calculate log probabilities for selected actions
        log_probs_list = []
        for i, (action_str, legal_moves) in enumerate(zip(batch_actions, batch_legal_moves)):
            action_idx = move_to_idx[action_str]
            
            # Create mask for legal moves
            mask = torch.zeros(4096, device=self.device)
            mask[legal_moves] = 1
            
            # Apply mask and get log probabilities
            masked_logits = logits[i].masked_fill(mask == 0, float('-inf'))
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            log_probs_list.append(log_probs[action_idx])
        
        log_probs_tensor = torch.stack(log_probs_list)
        
        # REINFORCE loss
        loss = -torch.mean(log_probs_tensor * returns_tensor)
        
        # Check for NaN or inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
            return 0.0, []
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Calculate DTMs for logging
        dtms = []
        for episode in episodes_data:
            if episode['done'] and episode['rewards'] and episode['rewards'][-1] > 0:
                dtm = 2 * (len(episode['states']) - 0.5)
                dtms.append(dtm)
        
        return loss.item(), dtms
    
    def train(self, endgames: List[str]):
        """
        Train the policy using batch processing.
        """
        n_episodes = len(endgames)
        n_batches = int(n_episodes / self.batch_size)
        
        all_losses = []
        all_dtms = []
        
        # Create progress bar for batches
        pbar = tqdm(range(n_batches), desc="Training")
        
        for batch_idx in pbar:
            # Get batch of endgames
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_episodes)
            batch_endgames = endgames[start_idx:end_idx]
            
            # Create environments for batch
            envs = [Env.from_fen(fen, defender=self.defender) for fen in batch_endgames]
            
            # Sample episodes in batch
            episodes_data = self.sample_episodes(envs)
            
            # Train on batch
            loss, dtms = self.train_batch(episodes_data)
            
            all_losses.append(loss)
            all_rewards = [ep['rewards'] for ep in episodes_data]
            all_dtms.extend(dtms)

            # Update progress bar
            avg_loss = np.mean(all_losses) if all_losses else 0.0
            avg_reward = np.mean([r for rewards in all_rewards for r in rewards]) if all_rewards else 0.0
            avg_dtm = np.mean(all_dtms) if all_dtms else float('inf')
            
            pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'avg_reward': f'{avg_reward:.4f}'
            })
        
        # Save final model
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': all_losses,
            'rewards': all_rewards,
            'dtms': all_dtms
        }, config['model_path'])
        
        logger.info(f"Training completed. Final avg loss: {np.mean(all_losses):.4f}")
        
        return all_losses, all_rewards, all_dtms