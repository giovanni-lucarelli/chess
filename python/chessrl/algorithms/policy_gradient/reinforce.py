#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import json
import numpy as np
import torch 
from torch import nn 
import logging
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from chessrl.utils.load_config import load_config
from chessrl.utils.fen_parsing import parse_fen
from chessrl.utils.endgame_loader import sample_endgames, get_all_endgames_from_dtz
from typing import List
import random

import os
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# chess
from chessrl import Env, SyzygyDefender
from chessrl import chess_py as cp

class Policy(nn.Module):
    """
    Policy function. 
    Input:
        - Piece placement (tensor)
        - Whose turn it is (tensor)
    Output: 
        - Distribution over actions from that state
    """
    def __init__(self):
        super().__init__()
        
        # CNN layers
        self.board_conv = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense layers
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4096) # 4096 are the possible actions (for simple endgames)
        )

    def forward(self, fen):
        # fen can be either a string or already a tensor
        if isinstance(fen, str):
            x = parse_fen(fen).unsqueeze(0)  # → [1, 8, 8, 12]
        elif isinstance(fen, torch.Tensor):
            x = fen  # Already a tensor with batch dimension
        else:
            logger.error(f"Expected str or torch.Tensor, got {type(fen)}")
        x = x.permute(0, 3, 1, 2)  # → [batch, 12, 8, 8] for CNN
        x = self.board_conv(x)     # → [batch, 128, 8, 8] 
        x = self.global_pool(x)    # → [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)  # → [batch, 128]
        x = self.fc(x)             # → [batch, 4096]
        return x
    
    def get_action_probs(self, fen, legal_moves=None):
        """
        Get action probabilities, optionally masked for legal moves only.
        """
        logits = self.forward(fen)
        
        # Apply legal move mask if provided
        if legal_moves is not None:
            mask = torch.full_like(logits, float('-inf'))
            for move_idx in legal_moves:
                mask[0, move_idx] = 0
            logits = logits + mask
            
        return torch.softmax(logits, dim=-1)
    
    def get_log_probs(self, fen, legal_moves=None):
        """
        Get log probabilities for gradient computation.
        """
        logits = self.forward(fen)
        
        # Apply legal move mask if provided
        if legal_moves is not None:
            mask = torch.full_like(logits, float('-inf'))
            for move_idx in legal_moves:
                mask[0, move_idx] = 0
            masked_logits = logits + mask
            
            # For numerical stability, subtract max of legal moves before log_softmax
            legal_logits = logits[0, legal_moves]
            max_legal_logit = legal_logits.max()
            masked_logits = masked_logits - max_legal_logit
            
            return torch.log_softmax(masked_logits, dim=-1)
        else:
            return torch.log_softmax(logits, dim=-1)
    
    def predict(self, env, move_to_idx, epsilon=config['epsilon']):
        """
        Predict the best move for the given game state.
        
        Args:
            epsilon: If > 0, use epsilon-greedy exploration
        """
        # Convert game state to FEN
        fen = env.to_fen()

        # Get legal move indices
        legal_moves = []
        side_to_move = env.state().get_side_to_move()
        for move in env.state().legal_moves(side_to_move):
            move_str = cp.Move.to_uci(move)[:4]
            if move_str in move_to_idx:
                legal_moves.append((move_to_idx[move_str], move))
        
        if not legal_moves:
            logger.warning(f"No legal moves found for position: {fen}")
            return None
        
        # Get action probabilities
        fen_tensor = parse_fen(fen).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.get_action_probs(fen_tensor, [idx for idx, _ in legal_moves])
            action_probs = action_probs.squeeze(0).cpu().numpy()
        
        # Choose move (epsilon-greedy or greedy)
        legal_indices = [idx for idx, _ in legal_moves]
        legal_probs = action_probs[legal_indices]
        
        if epsilon > 0 and np.random.random() < epsilon:
            # Random exploration
            selected_idx = np.random.choice(len(legal_moves))
            best_legal_idx = legal_indices[selected_idx]
        else:
            # Greedy selection
            best_legal_idx = legal_indices[legal_probs.argmax()]
        
        logger.debug(f"Found {len(legal_moves)} legal moves, best prob: {legal_probs.max():.4f}")
        
        # Return the corresponding move
        for idx, move in legal_moves:
            if idx == best_legal_idx:
                return move
        
        return None
    
class REINFORCE:
    def __init__(
            self,
            lr: float = config['alpha'],
            gamma: float = config['gamma'],
            epsilon: float = config['epsilon'],
            max_steps: int = config['max_steps']
    ):
        self.lr = lr 
        self.gamma = gamma 
        self.epsilon = epsilon
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.dtm_history = []
        self.max_steps = max_steps
        
        # Initialize Syzygy defender for black moves
        self.defender = SyzygyDefender(tb_path='/Users/christianfaccio/UniTs/projects/chess/syzygy-tables')

        # For converting between moves and indices
        self.move_to_idx = {}
        self.idx_to_move = {}
        self._build_move_mappings()

    def _build_move_mappings(self):
        """
        Build mappings between chess moves and action indices.
        Using custom square naming to match C++ implementation.
        """
        idx = 0
        
        # Generate square names manually (a1, b1, ..., h8)
        files = "abcdefgh"
        ranks = "12345678"
        squares = [f + r for r in ranks for f in files]
        
        # Generate all possible moves (from_square, to_square)
        for from_sq in squares:
            for to_sq in squares:
                if from_sq != to_sq:
                    move_str = from_sq + to_sq
                    self.move_to_idx[move_str] = idx
                    self.idx_to_move[idx] = move_str
                    idx += 1

    def get_legal_move_indices(self, game):
        """
        Get indices of legal moves for the current position.
        """
        legal_moves = []
        side_to_move = game.get_side_to_move()
        
        # Get legal moves for the current side
        for move in game.legal_moves(side_to_move): 
            move_str = cp.Move.to_uci(move)[:4]  
            if move_str in self.move_to_idx:
                legal_moves.append(self.move_to_idx[move_str])
        return legal_moves
    
    def sample_episode(self, fen: str, max_steps: int = config['max_steps']):
        """
        Sample a trajectory using the current policy for White
        and oracle (best possible move) for black.
        Returns simplified format for easier processing.
        """
        env = Env.from_fen(
            fen,
            step_penalty=0.01,
            defender=self.defender,   
        )
        game = env.state()
        
        # Store episode data
        states = []      # FEN strings
        actions = []     # Move indices
        rewards = []     # Immediate rewards
        players = []     # Which player made the move (0=white, 1=black)
        
        for step in range(max_steps):
            current_fen = game.to_fen()
            side_to_move = game.get_side_to_move()
            
            if side_to_move == cp.Color.WHITE:  # White turn - use policy
                legal_moves = self.get_legal_move_indices(game)
                
                if not legal_moves:
                    break  # No legal moves available
                
                # Get action probabilities
                fen_tensor = parse_fen(current_fen).unsqueeze(0)
                with torch.no_grad():
                    action_probs = self.policy.get_action_probs(fen_tensor, legal_moves)
                    action_probs = action_probs.squeeze(0).cpu().numpy()
                
                # Extract probabilities for legal moves only
                legal_probs = action_probs[legal_moves]  # Remove the extra [0,] indexing
                # Add small epsilon to avoid zero probabilities
                legal_probs = legal_probs + 1e-8
                # Normalize to ensure they sum to 1
                legal_probs = legal_probs / legal_probs.sum()
                
                # Sample from legal moves
                selected_legal_idx = np.random.choice(len(legal_moves), p=legal_probs)
                action_idx = legal_moves[selected_legal_idx]
                
                # Convert to move
                move_str = self.idx_to_move[action_idx]
                move = cp.Move.from_strings(game, move_str[:2], move_str[2:4])
                
                # Take step
                step_result = env.step(move)

                # Store data
                states.append(current_fen)
                actions.append(action_idx)
                rewards.append(step_result.reward)
                players.append(cp.Color.WHITE)  
            
            if game.is_game_over():
                break
        
        # Calculate DTM (Distance to Mate) only if checkmate achieved (positive reward)
        # DTM should count white moves only when the game ends in mate
        if step_result.reward > 0:
            DTM = len([p for p in players if p == cp.Color.WHITE])  # Count white moves to mate
        else:
            DTM = float('inf')  # No mate achieved

        # Cap max steps and give draw penalty if not checkmate
        if not step_result.done:
            rewards[-1] = -0.5
        
        return states, actions, rewards, players, DTM
    
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

    def train_episode(self, starting_fen: str):
        """
        Train on a single episode (legacy method for backward compatibility).
        """
        states, actions, rewards, players, DTM = self.sample_episode(starting_fen)
        
        if not states:
            logger.info("No moves made in episode")
            return None, DTM
        
        # Calculate returns for each time step
        returns = self.calculate_returns(rewards)
        
        # Convert episode data to tensors and train
        log_probs = []
        episode_returns = []
        
        for step in range(len(states)):
            fen_tensor = parse_fen(states[step]).unsqueeze(0)
            
            # Create temporary game to get legal moves
            temp_game = cp.Game()
            temp_game.reset_from_fen(states[step])
            legal_moves = self.get_legal_move_indices(temp_game)
            
            # Get log probabilities
            log_prob_dist = self.policy.get_log_probs(fen_tensor, legal_moves)
            log_prob = log_prob_dist[0, actions[step]]
            
            log_probs.append(log_prob)
            episode_returns.append(returns[step])
        
        # Calculate loss with numerical stability
        log_probs_tensor = torch.stack(log_probs)
        returns_tensor = torch.tensor(episode_returns, dtype=torch.float32)
        
        # REINFORCE loss: -E[log π(a|s) * G]
        loss = -torch.mean(log_probs_tensor * returns_tensor)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), DTM

    def train(self, 
              n_episodes: int = config['n_episodes']):
        """
        Train the policy on multiple endgame positions with batch updates.
        
        Args:
            endgame: List of endgame FENs to sample from
            n_episodes: Total number of episodes to run
            episodes_per_update: Number of episodes to collect before each parameter update
        """
        endgames = get_all_endgames_from_dtz(csv_path='../../../../syzygy-tables/krk_dtz.csv', dtz=3) # training on KRvK with DTZ=3 only

        accumulated_dtm = []
        accumulated_loss = []

        # Create progress bar
        with tqdm(total=n_episodes, desc="Training Episodes") as pbar:
            for episode in range(n_episodes):
                endgame_data = random.choice(endgames)
                starting_fen = endgame_data['fen']  # Extract FEN string from dictionary
                loss, DTM = self.train_episode(starting_fen)

                accumulated_loss.append(loss)

                # Store DTM for plotting
                if DTM is not None:
                    accumulated_dtm.append(DTM)
                
                # Update progress bar with recent metrics
                recent_loss = np.mean(accumulated_loss[-100:]) if len(accumulated_loss) >= 100 else np.mean(accumulated_loss)
                recent_dtm = np.mean(accumulated_dtm[-100:]) if len(accumulated_dtm) >= 100 else (np.mean(accumulated_dtm) if accumulated_dtm else float('inf'))
                
                pbar.set_postfix({
                    'Loss': f'{recent_loss:.3f}' if recent_loss is not None else 'N/A',
                    'DTM': f'{recent_dtm:.1f}' if recent_dtm != float('inf') else 'inf'
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
