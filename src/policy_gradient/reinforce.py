#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import json
import numpy as np
import torch # type: ignore
from torch import nn # type: ignore
import logging
from utils.fen_parsing import *
from utils.load_config import load_config

# chess
from build.chess_py import Game, Env

def load_config(config_path = 'config.json'):
    logger.info('Loading config file...')
    with open(config_path, 'r') as f:
        return json.load(f)
    
config = load_config()
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class policy(nn.Module):
    """
    Policy function. 
    It is a neural network with input FEN decomposed in:
        - Piece placement
        - Active color
        - Castling rights
        - En passant
        - Halfmove clock (not considered)
        - Fullmove number (not considered)
    and output:
        - probability distribution over possible actions (4096)
    """
    def __init__(self):
        super().__init__()
        # Board processing (CNN for spatial patterns)
        self.board_conv = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Game state processing
        self.game_state_fc = nn.Sequential(
            nn.Linear(71, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Combined processing
        self.combined_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 4096)  # All possible moves
        )

    def forward(self, fen):
        board_tensor, game_state = parse_fen_to_features(fen)
        board_tensor = board_tensor.permute(2, 0, 1) # Transpose from [8, 8, 12] to [12, 8, 8] for CNN
        board_features = self.board_conv(board_tensor.unsqueeze(0)) # add batch dim
        board_flat = board_features.view(-1) # flatten
        game_features = self.game_state_fc(game_state)
        combined = torch.cat([board_flat, game_features])
        logits = self.combined_fc(combined)
        return logits
    
    def predict(self, game_state):
        """
        Predict the best move for the given game state.
        Returns the Move object that the policy thinks is best.
        """
        fen = game_state.to_fen()
        
        # Get legal moves for current player
        legal_moves = game_state.legal_moves(game_state.get_side_to_move())
        if len(legal_moves) == 0:
            logger.info('!!! GAME OVER !!!')
            return None  # Game over
        
        # Convert legal moves to action indices
        legal_actions = []
        legal_moves_list = []
        for move in legal_moves:
            action = int(move.from_square) * 64 + int(move.to_square)
            legal_actions.append(action)
            legal_moves_list.append(move)
        
        # Get policy probabilities and mask illegal actions
        with torch.no_grad():  # No gradients needed for prediction
            logits = self.forward(fen)
            probs = torch.softmax(logits, dim=-1)
            
            # Create masked probabilities (only legal actions)
            legal_probs = torch.zeros_like(probs)
            for action in legal_actions:
                legal_probs[action] = probs[action]
            
            # Find the action with highest probability
            if legal_probs.sum() > 0:
                best_action_idx = torch.argmax(legal_probs).item()
            else:
                # Fallback: choose first legal move
                best_action_idx = legal_actions[0]
            
            # Find the corresponding move
            if best_action_idx in legal_actions:
                move_idx = legal_actions.index(best_action_idx)
                return legal_moves_list[move_idx]
            else:
                # Fallback: return first legal move
                return legal_moves_list[0]

class REINFORCE:
    def __init__(
            self,
            n_episodes=config['n_episodes'],
            max_steps=config['max_steps'],
            lr=config['lr'],
            step_penalty=config['step_penalty'],
            gamma=config['gamma']
    ):
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.gamma = gamma
        self.policy = policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def sample_episode(self, fen, policy):
        """
        Function that sample an episode using the current policy.
        Inputs:
            - initial board state (fen)
            - current policy
        The output should be a List of tuples:
            episode = [(s_t,a_t,r_{t+1})]
        """
        logger.info('Sampling episode...')
        episode = []
        game = Game()
        game.reset_from_fen(fen)
        environment = Env(game, gamma = self.gamma, step_penalty = self.step_penalty)
        for time_step in range(self.max_steps):
            logger.info(f'Time step number {time_step} / {self.max_steps}')
            state = environment.state().to_fen()
            
            # Get legal moves for current player
            legal_moves = environment.state().legal_moves(environment.state().get_side_to_move())
            if len(legal_moves) == 0:
                logger.info('!!! GAME OVER !!!')
                break  # Game over (checkmate or stalemate)
            
            # Convert legal moves to action indices
            legal_actions = []
            legal_moves_list = []
            for move in legal_moves:
                action = int(move.from_square) * 64 + int(move.to_square)  # Convert move to action index
                legal_actions.append(action)
                legal_moves_list.append(move)
            
            # Get policy probabilities and mask illegal actions
            logits = policy(environment.state().to_fen())
            probs = torch.softmax(logits, dim=-1)
            
            # Create masked probabilities (only legal actions have non-zero probability)
            legal_probs = torch.zeros_like(probs)
            for action in legal_actions:
                legal_probs[action] = probs[action]
            
            # Renormalize legal probabilities
            if legal_probs.sum() > 0:
                legal_probs = legal_probs / legal_probs.sum()
            else:
                # Fallback: uniform distribution over legal actions
                logger.info('Fallback: using uniform distribution')
                legal_probs = torch.zeros_like(probs)
                for action in legal_actions:
                    legal_probs[action] = 1.0 / len(legal_actions)
            
            # Sample from legal actions only
            logger.info('Sampling...')
            action = torch.multinomial(legal_probs, 1).item()
            
            # Find the corresponding move (we already validated it's legal)
            move_idx = legal_actions.index(action)
            move = legal_moves_list[move_idx]
            
            step_result = environment.step(move) 
            reward = step_result.reward
            episode.append((state,action,reward))
            if step_result.done == True:
                logger.info(f'Stopping episode at time step {time_step}')
                break
        return episode


    def calculate_return(self, episode, step):
        """
        Function that calculates the cumulative reward of a step.
        """
        logger.info('Calculating return...')
        reward = 0
        gamma = self.gamma
        for t in range(len(episode) - step):
            reward += gamma**t * episode[step + t][2]
        return reward
    
    def calculate_loss(self, state, action, return_value):
        """
        Loss Function
        """
        logger.info('Calculating loss...')
        logits = self.policy(state)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_prob_action = log_probs[action]
        loss = -log_prob_action * return_value
        return loss

    def train(self, fen):
        logger.info('Starting training...')
        for n in range(self.n_episodes):
            logger.debug(f'Episode number {n}')
            episode = self.sample_episode(fen, self.policy) 
            for step in range(len(episode)):
                logger.debug(f'Step number {step} of episode {n}')
                state = episode[step][0] # take state at time t 
                action = episode[step][1] # take action at time t
                return_value = self.calculate_return(episode, step) # calculate return value (v_t)
                loss = self.calculate_loss(state, action, return_value)
                self.optimizer.zero_grad() # clear previous gradients
                loss.backward() # calculate gradients
                self.optimizer.step() # update weights
        self.save_model()
        return self.policy
    
    def save_model(self, filepath=config['filepath']):
        """
        Save the trained policy weights to a file.
        """
        logger.info(f'Saving model with filepath {filepath}')
        torch.save(self.policy.state_dict(), filepath)
    
    def load_model(self, filepath=config['filepath']):
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
    def load_policy_for_prediction(filepath=config['log_level']):
        """
        Static method to load a trained policy for prediction only.
        Returns a policy instance ready for prediction.
        """
        logger.info(f'Loading model with filepath {filepath}')
        try:
            loaded_policy = policy()
            loaded_policy.load_state_dict(torch.load(filepath))
            loaded_policy.eval()
            return loaded_policy
        except FileNotFoundError:
            logger.error(f"File {filepath} not found. Returning randomly initialized policy.")
            return policy()
        except Exception as e:
            logger.error(f"Error loading policy: {e}")
            return policy()