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
from chessrl.utils.load_config import load_config
config = load_config()
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import pickle
import numpy as np
# chess
from chessrl.env import Env, SyzygyDefender
from build.chess_py import Move
from chessrl.utils.create_endgames import generate_all_endgame_positions, pieces_to_board_string, parse_fen_pieces

class ValueIteration:
    def __init__(
            self,
            save_path=config['savepath_value_iteration'],
            tolerance = config['tolerance'],
            step_penalty=config['step_penalty'],
            gamma=config['gamma'],
    ):
        self.step_penalty = step_penalty
        self.gamma = gamma
        self.save_path = save_path
        self.tolerance = tolerance


    def train(self,base_fen):
        """
        Trains value iteration on the chess endgame starting from base_fen.
        Returns a dictionary: state_tuple -> best_action
        """
        logger.info('Creating the states...')
        
        # Inizialize vector of values for all states
        states, state_to_idx, values = generate_all_endgame_positions(base_fen)
        newValues = values.copy()
        newPolicy = {}

        logger.info(f"Training on {len(states)} states")


        for i in range(5):
            logger.info(f"Starting iteration {i+1}/5")
            values = newValues.copy()
            
            # cycle over all the states, where each state is defined by the pieces position
            for state_idx, fen in enumerate(states):
                    if state_idx % 10000 == 0:  # Adjust frequency as needed
                        logger.info(f"  Processing state {state_idx+1}/{len(states)} in iteration {i+1}")
                        
                    maxvalue = -100
                    TB_PATH = "tablebase"  
                    defender = SyzygyDefender(TB_PATH)                  
                    
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
                        except KeyError:
                            logger.info(f"State {enviroment.state().to_fen()} not found in state space, previous {fen}, next move is for {enviroment.state().get_side_to_move()}")
                            break
                        
                        if (value_action > maxvalue):
                            maxvalue = value_action
                            bestact = A
        
                    # It stores the new value and policy of state S
                    newValues[state_to_idx[fen]] = maxvalue
                    newPolicy[fen] = bestact

            #Estimate change
            err = np.sqrt(np.mean( (newValues - values)**2))
            logger.info('Distance between V_{}(S) and V_{}(S) is: {}'.format(i, i+1, err))
            if err < self.tolerance:
                logger.info(f'Convergence reached with tolerance {self.tolerance} after {i+1} iterations.')
                break

        logger.info('Training completed.')
        logger.debug(f'Best policy found: {newPolicy}')

        # Save policy to file
        with open(self.save_path, "wb") as f:
            # Convert Move objects to UCI strings for pickling
            serializable_policy = {}
            for state, move in newPolicy.items():
                if move is not None:
                    serializable_policy[state] = Move.to_uci(move)
                else:
                    serializable_policy[state] = None
            
            pickle.dump(serializable_policy, f)

        return newPolicy

