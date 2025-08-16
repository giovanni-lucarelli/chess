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
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt # type: ignore
from utils.fen_parsing import *
from utils.load_config import load_config
from utils.create_endgames import generate_endgame_positions
import random

# chess
from build.chess_py import Game, Env
    
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
            gamma=config['gamma'],
            batch_size=config['batch_size']
    ):
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.gamma = gamma
        self.batch_size = batch_size
        self.baseline = 0.0  # Running average baseline
        self.baseline_decay = 0.95  # Decay factor for baseline update
        self.dtm_history = []  # Track DTM (steps to completion) over training
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
        logger.debug('Sampling episode...')
        episode = []
        game = Game()
        game.reset_from_fen(fen)
        environment = Env(game, gamma = self.gamma, step_penalty = self.step_penalty)
        steps_pbar = tqdm(range(self.max_steps), desc="Episode Steps", unit="step", leave=False)
        for time_step in steps_pbar:
            logger.debug(f'Time step number {time_step} / {self.max_steps}')
            state = environment.state().to_fen()
            
            # Get legal moves for current player
            legal_moves = environment.state().legal_moves(environment.state().get_side_to_move())
            if len(legal_moves) == 0:
                logger.info('!!! GAME OVER !!!')
                steps_pbar.close()
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
                logger.debug('Fallback: using uniform distribution')
                legal_probs = torch.zeros_like(probs)
                for action in legal_actions:
                    legal_probs[action] = 1.0 / len(legal_actions)
            
            # Sample from legal actions only
            logger.debug('Sampling...')
            action = torch.multinomial(legal_probs, 1).item()
            
            # Find the corresponding move (we already validated it's legal)
            move_idx = legal_actions.index(action)
            move = legal_moves_list[move_idx]
            
            step_result = environment.step(move) 
            reward = step_result.reward
            episode.append((state,action,reward))
            steps_pbar.set_postfix({"Step": time_step+1, "Reward": reward, "Legal_moves": len(legal_moves)})
            if step_result.done == True:
                logger.info(f'Stopping episode at time step {time_step}')
                steps_pbar.close()
                break
        steps_pbar.close()
        return episode

    def calculate_return(self, episode, step):
        """
        Function that calculates the cumulative reward of a step.
        """
        logger.debug('Calculating return...')
        reward = 0
        gamma = self.gamma
        for t in range(len(episode) - step):
            reward += gamma**t * episode[step + t][2]
        return reward
    
    def calculate_loss(self, state, action, return_value):
        """
        Loss Function using advantage (return - baseline)
        """
        logger.debug('Calculating loss...')
        logits = self.policy(state)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_prob_action = log_probs[action]
        advantage = return_value - self.baseline
        loss = -log_prob_action * advantage
        return loss

    def train(self, fen):
        logger.info('Starting training...')
        
        # Generate positions ONCE at start of training for efficiency
        logger.info('Generating endgame positions...')
        positions = generate_endgame_positions(fen, 100) # Generating 100 random (legal) positions from current endgame (maintaining same pieces, just permuting positions)
        logger.info(f'Generated {len(positions)} endgame positions for training')
        
        n_batches = self.n_episodes // self.batch_size
        batch_pbar = tqdm(range(n_batches), desc="Training Batches", unit="batch")
        
        for batch_idx in batch_pbar:
            logger.debug(f'Batch number {batch_idx}')
            
            # Collect multiple episodes for this batch
            all_episodes = []
            batch_steps = 0
            batch_dtm = []  # Track DTM (Distance to Mate) for this batch
            for episode_in_batch in range(self.batch_size):
                sample_fen = random.choice(positions) # sample random from the positions to ensure better generalisation for the endgame
                episode = self.sample_episode(sample_fen, self.policy)
                all_episodes.append(episode)
                batch_steps += len(episode)
                
                # Calculate DTM: if episode ends with mate, DTM = episode length
                # If episode times out, DTM = max_steps (failed to mate)
                episode_length = len(episode)
                if episode_length < self.max_steps:
                    # Episode ended early (likely mate or game over)
                    dtm = episode_length
                else:
                    # Episode hit time limit (failed to find mate)
                    dtm = self.max_steps
                batch_dtm.append(dtm)
            
            # Calculate returns for baseline update
            all_returns = []
            for episode in all_episodes:
                for step in range(len(episode)):
                    return_value = self.calculate_return(episode, step)
                    all_returns.append(return_value)
            
            # Update baseline with average return from this batch
            if len(all_returns) > 0:
                batch_avg_return = sum(all_returns) / len(all_returns)
                self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * batch_avg_return
            
            # Store DTM statistics for this batch
            avg_dtm = sum(batch_dtm) / len(batch_dtm) if batch_dtm else self.max_steps
            self.dtm_history.append(avg_dtm)
            
            # Calculate total loss across all episodes in batch
            total_loss = 0
            total_experiences = 0
            
            self.optimizer.zero_grad() # clear previous gradients
            
            for episode in all_episodes:
                for step in range(len(episode)):
                    state = episode[step][0] # take state at time t 
                    action = episode[step][1] # take action at time t
                    return_value = self.calculate_return(episode, step) # calculate return value (v_t)
                    loss = self.calculate_loss(state, action, return_value)
                    total_loss += loss
                    total_experiences += 1
            
            # Average the loss and backpropagate
            if total_experiences > 0:
                avg_loss = total_loss / total_experiences
                avg_loss.backward() # calculate gradients
                self.optimizer.step() # update weights
            
            batch_pbar.set_postfix({
                "Batch": batch_idx+1, 
                "Avg_DTM": f"{avg_dtm:.1f}",
                "Total_Exp": total_experiences,
                "Baseline": f"{self.baseline:.3f}"
            })
        
        batch_pbar.close()
        self.save_model()
        self.plot_dtm_progress()
        return self.policy
    
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