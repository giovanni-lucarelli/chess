#!/usr/bin/env python3 

# system
import os

# utils
import numpy as np
import logging 
from chessrl.utils.load_config import load_config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from chessrl.utils.endgame_loader import load_all_positions_with_actions
from chessrl.utils.io import save_policy_jsonl, save_values
from tqdm import tqdm

# chess
from chessrl.env import Env, SyzygyDefender
from chessrl import chess_py  

class TD_Control():
    def __init__(self,  
                gamma=config['gamma'], 
                alpha=config['alpha'], 
                epsilon = config['epsilon'],
                endgame_type=config['endgame_type']
                ):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.endgame_type = endgame_type
        endgame_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tablebase', self.endgame_type, f"{self.endgame_type}_full.csv")
        self.states, self.positions_actions_to_idx, self.Qvalues = load_all_positions_with_actions(endgame_path)
        self.defender = SyzygyDefender('../../../../tablebase/krk')

    
    # This is the only method that changes between Q-learning and SARSA
    def single_step_update(self, 
                           state, 
                           action, 
                           reward, 
                           new_state, 
                           new_legal_actions, 
                           done,
                           td_error_algorithm: str):
        """
        Performs a single TD-learning update step.
        state: current state 
        action: action taken, uci format
        reward: reward received 
        new_state: resulting state 
        new_legal_actions: list of legal actions in the new state
        done: whether the episode has ended
        """
        if done:
            td_target = reward
            next_action = None
        else:
            if td_error_algorithm == "QLearning":
                next_action = self.get_action_greedy(new_state, new_legal_actions)
            elif td_error_algorithm == "SARSA":
                next_action = self.get_action_epsilon_greedy(new_state, new_legal_actions)
            else:
                raise ValueError(f"Unknown TD algorithm: {td_error_algorithm}")
            
            if next_action is None:
                td_target = reward
            else:
                td_target = reward + self.gamma * self.Qvalues[self.positions_actions_to_idx[(new_state, chess_py.Move.to_uci(next_action))]]
        td_error = td_target - self.Qvalues[self.positions_actions_to_idx[(state, action)]]
        
        # Update Q-value
        self.Qvalues[self.positions_actions_to_idx[(state, action)]] += self.alpha * td_error

        return next_action

    def get_action_greedy(self, state, legal_moves):
        """
        Selects an action using greedy policy.
        state: current state
        legal_moves: list of legal actions in the current state
        Returns: selected action or, if there are no legal moves, None
        """
        if len(legal_moves)==0 or legal_moves is None:
            return None
        
        best_value = -np.inf
        best_move = None
        for move in legal_moves:
            idx = self.positions_actions_to_idx[(state, chess_py.Move.to_uci(move))]
            q_val = self.Qvalues[idx]

            if q_val > best_value:
                best_value = q_val
                best_move = move
        return best_move

    def get_action_epsilon_greedy(self, state, legal_moves):
        """
        Selects an action using epsilon-greedy policy.
        state: current state
        legal_moves: list of legal actions in the current state
        Returns: selected action or, if there are no legal moves, None
        """
        if len(legal_moves)==0 or legal_moves is None:
            return None
        
        epsilon = self.epsilon
        if np.random.rand() < epsilon:
            random_move = np.random.randint(len(legal_moves))
            return legal_moves[random_move]  # Explore
        else:
            best_value = -np.inf
            best_move = None
            for move in legal_moves:
                idx = self.positions_actions_to_idx[(state, chess_py.Move.to_uci(move))]
                q_val = self.Qvalues[idx]

                if q_val > best_value:
                    best_value = q_val
                    best_move = move  # Exploit
            return best_move
    

    def train(self, endgames, td_error_algorithm: str, n_episodes: int = config['n_episodes']):
        performance_traj_Q = np.zeros(n_episodes)
        
        logger.info(f'Starting {td_error_algorithm} training...')
        with tqdm(total=len(endgames), desc="Training") as pbar:  
            for endgame in endgames:
                env = Env.from_fen(endgame, gamma = self.gamma, defender=self.defender) 
                a = self.get_action_epsilon_greedy(env.state().to_fen(), env.state().legal_moves(env.state().get_side_to_move()))
                s = endgame
                
                # Skip if no legal moves available
                if a is None:
                    logger.debug(f"No legal moves available for position {s}")
                    continue
                
                counter = 0 
                while True:
                    if counter >= config['max_steps']:
                        logger.debug(f"Reached max steps of {config['max_steps']}, ending episode.")
                        break

                    # Evolve one step
                    step_result = env.step(a)
                    new_s = env.to_fen()
                    done = step_result.done 

                    r = step_result.reward
                    performance_traj_Q[counter] += r
                    
                    if (env.state().is_game_over()):
                        new_actions = []
                    else:
                        new_actions = env.state().legal_moves(env.state().get_side_to_move())
                    
                    # Single update with (S, A, R', S')
                    new_a = self.single_step_update(s, chess_py.Move.to_uci(a), r, new_s, new_actions, done, td_error_algorithm=td_error_algorithm)
                    
                    if done or new_a is None:
                        break
                        
                    a = new_a
                    s = new_s
                    counter += 1
                pbar.update(1)
        
        policy = {}
        
        logger.info(f'Starting saving best policy...')
        
        for state in self.states:
            env = Env.from_fen(state)
            legal_moves = env.state().legal_moves(env.state().get_side_to_move())
            best_value = -np.inf
            best_move = None
            for move in legal_moves:
                uci_move = chess_py.Move.to_uci(move)
                idx = self.positions_actions_to_idx[(state, uci_move)]
                q_val = self.Qvalues[idx]
                if q_val > best_value:
                    best_value = q_val
                    best_move = uci_move
            policy[state] = best_move
        
        # Save policy to file
        save_policy_jsonl(policy, f"../../../../artifacts/policies/TD_{td_error_algorithm}_{self.endgame_type}_greedy.jsonl")

        logger.info('Training completed.')

