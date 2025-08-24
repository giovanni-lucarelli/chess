# Chess as an MDP:
"""
__State__ S: all legal positions.

__Action__ A: All legal moves.

__Transition__ p: Deterministic.

p(S' | S, A) = 
- 1  _if    S' == S+A     and S' is allowed
- 0  _else_

__Reward__ R: Only when checkmate happens (=terminal state!): +1 if it's the player checkmate, -1 if it's the "enviroment" checkmate.
 """
import logging 
import os
from chessrl.utils.load_config import load_config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import pickle
import numpy as np
# chess
from chessrl.env import Env, SyzygyDefender
from chessrl.chess_py import Move
from chessrl.utils.endgame_loader import load_all_positions
from chessrl.utils.io import save_policy_jsonl, save_values


class ValueIteration:
    def __init__(
            self,
            tolerance = config['tolerance'],
            n_iterations = config['n_iterations'],
            step_penalty=config['step_penalty'],
            gamma=config['gamma'],
    ):
        self.step_penalty = step_penalty
        self.n_iterations = n_iterations
        self.gamma = gamma
        self.tolerance = tolerance


    def train(self,base_fen):
        """
        Trains value iteration on the chess endgame starting from base_fen.
        Returns a dictionary: state_tuple -> best_action
        """
        logger.info('Creating the states...')
        
        # Inizialize vector of values for all states
        #states, state_to_idx, values = generate_all_endgame_positions(base_fen)
        endgame_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'syzygy-tables', config['endgame_type'] + '_dtz.csv')
        states, state_to_idx, values =load_all_positions(endgame_path)

        newValues = values.copy()
        newPolicy = {}

        logger.info(f"Training on {len(states)} states")

        n_iterations = self.n_iterations

        TB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'syzygy-tables')
        defender = SyzygyDefender(TB_PATH)

        os.makedirs("artifacts/policies", exist_ok=True)
        os.makedirs("artifacts/values", exist_ok=True)

        for i in range(n_iterations):
            logger.info(f"Starting iteration {i+1}/{n_iterations}...")
            values = newValues.copy()
            
            # cycle over all the states, where each state is defined by the pieces position
            for state_idx, fen in enumerate(states):
                    if state_idx % 10000 == 0:  # Adjust frequency as needed
                        logger.info(f"  Processing state {state_idx+1}/{len(states)} in iteration {i+1}")
                        
                    maxvalue = -100
                    
                    enviroment = Env.from_fen(fen, gamma = self.gamma, step_penalty = self.step_penalty, defender=defender)
                    color = enviroment.state().get_side_to_move()

                    # Check if it's already checkmate
                    if (enviroment.is_terminal()):
                        values[state_to_idx[fen]]= 0
                        newPolicy[fen] = None
                        continue

                    bestact = enviroment.state().legal_moves(color)[0] 

                    # it "tries out" all actions and store the best
                    for A in enviroment.state().legal_moves(color):
                        enviroment = Env.from_fen(fen, gamma = self.gamma, step_penalty = self.step_penalty, defender = defender)
                        # Contains both my move and the defender's reply
                        stepResult = enviroment.step(A)
                        R = stepResult.reward
                        
                        fen_new = enviroment.state().to_fen()
                        
                        try: 
                            value_action = R + values[state_to_idx[fen_new]]
                        except KeyError as e:
                            logger.info(f"State {enviroment.state().to_fen()} not found in state space, previous {fen}, next move is for {enviroment.state().get_side_to_move()}")
                            break
                        
                        if (value_action > maxvalue):
                            maxvalue = value_action
                            bestact = A
        
                    # It stores the new value and policy of state S
                    newValues[state_to_idx[fen]] = maxvalue
                    newPolicy[fen] = Move.to_uci(bestact)

            #Estimate change
            err = np.sqrt(np.mean( (newValues - values)**2))
            logger.info('Distance between V_{}(S) and V_{}(S) is: {}'.format(i, i+1, err))
            if err < self.tolerance:
                logger.info(f'Convergence reached with tolerance {self.tolerance} after {i+1} iterations.')
                break

            save_policy_jsonl(newPolicy, f"artifacts/policies/vi_krk_greedy_intermediate_{i+1}.jsonl")
            save_values(states, newValues, f"artifacts/values/vi_krk_greedy_intermediate_{i+1}.parquet")

        logger.info('Training completed.')
        
        # newPolicy : Dict[fen, uci]  |  states : List[fen]  |  newValues : np.ndarray
        save_policy_jsonl(newPolicy, "artifacts/policies/vi_krk_greedy.jsonl")
        save_values(states, newValues, "artifacts/values/vi_krk_values.parquet")

        return newPolicy
