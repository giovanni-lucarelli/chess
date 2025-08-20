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
from utils.fen_parsing import *
from utils.load_config import load_config
from utils.create_endgames import generate_endgame_positions
from utils.opponent_move import get_black_move
import random
import requests
from typing import List

config = load_config()
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# chess
from build.chess_py import Game, Env, Move, Color # type: ignore

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
            raise ValueError(f"Expected str or torch.Tensor, got {type(fen)}")
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
            logits = logits + mask
            
        return torch.log_softmax(logits, dim=-1)
    
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
            move_str = Move.to_uci(move)[:4]  
            if move_str in self.move_to_idx:
                legal_moves.append(self.move_to_idx[move_str])
        return legal_moves

    def sample_episode(self, starting_fen: str, max_steps: int = 200):
        """
        Sample a trajectory using the current policy for White
        and oracle (best possible move) for black.
        Returns simplified format for easier processing.
        """
        game = Game()
        game.reset_from_fen(starting_fen)
        env = Env(game, step_penalty=1)
        
        # Store episode data
        states = []      # FEN strings
        actions = []     # Move indices
        rewards = []     # Immediate rewards
        players = []     # Which player made the move (0=white, 1=black)
        
        for step in range(max_steps):
            current_fen = env.state().to_fen()
            side_to_move = game.get_side_to_move()
            
            if side_to_move == Color.WHITE:  # White turn - use policy
                legal_moves = self.get_legal_move_indices(game)
                logger.info(f'Found {len(legal_moves)} legal moves.')
                
                # Get action probabilities
                fen_tensor = parse_fen(current_fen).unsqueeze(0)  # Add batch dimension
                with torch.no_grad():
                    action_probs = self.policy.get_action_probs(fen_tensor, legal_moves)
                    action_probs = action_probs.squeeze(0).cpu().numpy()
                
                if not legal_moves:
                    break  # No legal moves available
                
                # Extract probabilities for legal moves only
                legal_probs = action_probs[legal_moves]
                # Normalize to ensure they sum to 1
                legal_probs = legal_probs / legal_probs.sum()
                logger.info(f'Legal probs sum: {legal_probs.sum()}')
                
                # Sample from legal moves
                selected_legal_idx = np.random.choice(len(legal_moves), p=legal_probs)
                action_idx = legal_moves[selected_legal_idx]
                
                # Convert to move
                move_str = self.idx_to_move[action_idx]
                logger.info(f'Move string: {move_str}')
                move = Move.from_strings(game, move_str[:2], move_str[2:4])
                
                # Take step
                step_result = env.step(move)
                game.do_move(move)

                # Reward
                reward = step_result.reward
                
                # Store data
                states.append(current_fen)
                actions.append(action_idx)
                rewards.append(reward)
                players.append(0)
                
            else:  # Black turn - use tablebase
                black_move_uci = get_black_move(current_fen)
                if black_move_uci is None:
                    break  # No legal moves or API error
                    
                move = Move.from_uci(game, black_move_uci)
                step_result = env.step(move)
                game.do_move(move)
                
                # Store data (we don't train on black moves, but need for reward calculation)
                states.append(current_fen)
                actions.append(-1)  # Dummy action for black
                rewards.append(step_result.reward)
                players.append(1)
            
            # Check if game is over
            if step_result.done:
                break
        
        # capping max steps and giving draw if not checkmate before the end
        if not step_result.done:
            rewards[-1] = -1000
        
        return states, actions, rewards, players
    
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
        Train on a single episode.
        """
        states, actions, rewards, players = self.sample_episode(starting_fen)

        DTM = len(states)
        
        if not states:
            logger.info("No moves made in episode")
            return 0.0  # No moves made
        
        # Calculate returns for each time step
        returns = self.calculate_returns(rewards)
        
        # Collect training data for white moves only
        log_probs = []
        episode_returns = []
        
        for i, player in enumerate(players):
            if player == 0:  # White move (stored as 0 in our tracking)
                fen_tensor = parse_fen(states[i]).unsqueeze(0)
                legal_moves = self.get_legal_move_indices_from_fen(states[i])  
                
                # Get log probabilities
                log_prob_dist = self.policy.get_log_probs(fen_tensor, legal_moves)
                log_prob = log_prob_dist[0, actions[i]]
                
                log_probs.append(log_prob)
                episode_returns.append(returns[i])
        
        if not log_probs:
            logger.info(f"No white moves to train on. Total states: {len(states)}, Players: {players}")
            return 0.0  # No white moves to train on
        
        # Calculate loss
        log_probs_tensor = torch.stack(log_probs)
        returns_tensor = torch.tensor(episode_returns, dtype=torch.float32)
        
        # REINFORCE loss: -E[log π(a|s) * G]
        loss = -torch.mean(log_probs_tensor * returns_tensor)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), DTM
    
    def get_legal_move_indices_from_fen(self, fen: str) -> List[int]:
        """
        Helper to get legal move indices from FEN string.
        """
        temp_game = Game()
        temp_game.reset_from_fen(fen)
        return self.get_legal_move_indices(temp_game)
    
    def train(self, endgames: List[str] = config['endgames'], n_episodes: int = config['n_episodes']):
        """
        Train the policy on multiple endgame positions.
        """
        for episode in range(n_episodes):
            # Randomly select an endgame position
            starting_fen = np.random.choice(endgames)
            
            # Train on this episode
            loss, DTM = self.train_episode(starting_fen)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{n_episodes}, Loss: {loss:.4f}, DTM: {DTM}")
        
        self.save_model()
                
    
    def plot_dtm_progress(self):
        """
        Plot DTM (Distance to Mate) progress over training batches
        """
        if not self.dtm_history:
            logger.warning("No DTM history to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.dtm_history) + 1), self.dtm_history, 'b-', linewidth=2)
        plt.xlabel('Training Batch')
        plt.ylabel('Average DTM (Distance to Mate)')
        plt.title('DTM Progress During Training')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line for optimal DTM (3 for the simple endgame)
        optimal_dtm = 3  # Known optimal for the simple endgame
        plt.axhline(y=optimal_dtm, color='r', linestyle='--', alpha=0.7, label=f'Optimal DTM = {optimal_dtm}')
        plt.legend()
        
        # Set y-axis to show reasonable range
        plt.ylim(0, min(self.max_steps, max(self.dtm_history) * 1.1))
        
        plt.tight_layout()
        plt.savefig('output/dtm_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"DTM progress plot saved to output/dtm_progress.png")
    
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
    
    @staticmethod
    def load_policy_for_prediction(filepath=config['filepath_test']):
        """
        Static method to load a trained policy for prediction only.
        Returns a policy instance ready for prediction.
        """
        logger.info(f'Loading model with filepath {filepath}')
        try:
            loaded_policy = Policy()
            loaded_policy.load_state_dict(torch.load(filepath))
            loaded_policy.eval()
            return loaded_policy
        except FileNotFoundError:
            logger.error(f"File {filepath} not found. Returning randomly initialized policy.")
            return Policy()
        except Exception as e:
            logger.error(f"Error loading policy: {e}")
            return Policy()