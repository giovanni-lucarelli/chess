# Q-Learning Control
import numpy as np
import logging 
import os
from chessrl.utils.load_config import load_config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import pickle
from chessrl.utils.endgame_loader import load_all_positions_with_actions
from chessrl.env import Env, SyzygyDefender
from chessrl import chess_py  # Import chess_py module to access Move class

class TD_Control():
    def __init__(self, type_of_endgame, td_error_algorithm="QLearning", n_episodes = 2000, n_starting_positions = 10, max_steps = 50, gamma=0.9, alpha=0.1, epsilon = 0.15, save_path = "output/"):
        self.save_path = save_path + td_error_algorithm + "_policy.pkl"
        self.starting_positions = n_starting_positions
        self.td_error_algorithm = td_error_algorithm 
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        endgame_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'syzygy-tables', type_of_endgame + '_dtz.csv')
        self.states, self.positions_actions_to_idx, self.Qvalues = load_all_positions_with_actions(endgame_path)

    
    # This is the only method that changes between Q-learning and SARSA
    def single_step_update(self, state, action, reward, new_state, new_legal_actions, done):
        """
        Performs a single TD-learning update step.
        state: current state 
        action: action taken, uci format
        reward: reward received 
        new_state: resulting state 
        new_legal_actions: list of legal actions in the new state
        done: whether the episode has ended
        """
        if self.td_error_algorithm == "QLearning":
            next_action = self.get_action_greedy(new_state, new_legal_actions)
        elif self.td_error_algorithm == "SARSA":
            next_action = self.get_action_epsilon_greedy(new_state, new_legal_actions)

        td_target = reward + (0 if done else self.gamma * self.Qvalues[self.positions_actions_to_idx[(new_state, chess_py.Move.to_uci(next_action))]])
        td_error = td_target - self.Qvalues[self.positions_actions_to_idx[(state, action)]]
        
        # Update Q-value
        self.Qvalues[self.positions_actions_to_idx[(state, action)]] += self.alpha * td_error

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
    
    def save_greedy_policy(self):
        """
        Derives the greedy policy from the learned Q-values.
        Returns: policy dictionary mapping states to best actions
        """
        policy = {}
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
        with open(self.save_path, "wb") as f:
             pickle.dump(policy, f)

    def train(self):
        performance_traj_Q = np.zeros(self.n_episodes)
        
        logger.info('Starting Q-learning training...')
        # Generate possible starting positions
        random_indices = np.random.choice(len(self.states), size=self.starting_positions, replace=False)
        random_starts = [self.states[i] for i in random_indices]
        # RUN OVER EPISODES

        for i in range(self.n_episodes):
            if i % 100 == 0:
                logger.info(f'Starting episode {i+1}/{self.n_episodes}')
            random_start = random_starts[i%self.starting_positions]

            TB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'syzygy-tables')
            defender = SyzygyDefender(TB_PATH)
            env = Env.from_fen(random_start, gamma = self.gamma, defender=defender) 
            legal_moves = env.state().legal_moves(env.state().get_side_to_move())
            a = self.get_action_epsilon_greedy(random_start, legal_moves)
            s = random_start
            
            done = False
            turns = 0 
            while not done:
                if turns >= self.max_steps:
                    break
                turns += 1

                # Evolve one step
                step_result = env.step(a)
                new_s = env.to_fen()
                done = step_result.done 

                r = step_result.reward
                # Keeps track of performance for each episode
                performance_traj_Q[i] += r
                
                if (env.state().is_game_over()):
                    new_actions = []
                else:
                    new_actions = env.state().legal_moves(env.state().get_side_to_move())

                # Choose new action index
                new_a = self.get_action_epsilon_greedy(new_s, new_actions)
                
                # Single update with (S, A, R', S')
                self.single_step_update(s, chess_py.Move.to_uci(a), r, new_s, new_actions, done)
                
                a = new_a
                s = new_s
        
        logger.info('Training completed.')

