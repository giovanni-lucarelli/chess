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
from utils.load_config import load_config
config = load_config()
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import pickle
import numpy as np
import chess
import chess.syzygy
import requests
# chess
from build.chess_py import Game, Env, Move
from utils.create_endgames import generate_all_endgame_positions, pieces_to_board_string, parse_fen_pieces


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


    def make_fen(self,base_fen, s, side="w") -> str:
        """Return the corresponding FEN string."""
        # Parse the base FEN
        fen_parts = base_fen.split()

        # Create new FEN
        new_board = pieces_to_board_string(s)
            
        # Keep original game state info (active color, castling, etc.)
        new_fen = new_board + ' ' + ' '.join(fen_parts[1:])
        
        # Debug logging
        logger.debug(f"make_fen: base_fen={base_fen}")
        logger.debug(f"make_fen: state s={s}")
        logger.debug(f"make_fen: generated new_fen={new_fen}")

        return new_fen

    def get_black_move(self, fen):
        """Query local Syzygy tablebase"""
        TB_PATH = "tablebase"
        board = chess.Board(fen)
        
        try:
            with chess.syzygy.open_tablebase(TB_PATH) as tablebase:
                # Query DTZ tablebase for best move
                best_move = None
                best_dtz = None
                
                for move in board.legal_moves:
                    board.push(move)
                    try:
                        dtz = tablebase.probe_dtz(board)  # distance to zeroing move
                        
                        if best_dtz is None or (dtz is not None and dtz < best_dtz):
                            best_dtz = dtz
                            best_move = move
                    except:
                        pass  # Skip moves that can't be evaluated
                    finally:
                        board.pop()
                
                return best_move.uci() if best_move else None
        except Exception as e:
            logger.error(f"Error accessing tablebase: {e}")
            return None

    def train(self,base_fen):
        """
        Trains value iteration on the chess endgame starting from base_fen.
        Returns a dictionary: state_tuple -> best_action
        """
        logger.info('Starting training...')
        # Generate positions ONCE at start of training for efficiency
        #logger.info('Generating endgame positions...')
        #positions = generate_endgame_positions(fen, 100) # Generating 100 random (legal) positions from current endgame (maintaining same pieces, just permuting positions)
        #logger.debug(f'Generated {len(positions)} endgame positions for training')
        
        # Inizialize vector of values for all states
        states, state_to_idx, values = generate_all_endgame_positions(base_fen)
        newValues = values.copy()
        newPolicy = {}

        logger.info(f"Training on {len(states)} states")


        for i in range(1):
            logger.info(f"Starting iteration {i+1}/100")
            values = newValues.copy()
            
            # cycle over all the states, where each state is defined by the pieces position
            for state_idx, s in enumerate(states):
                    if state_idx % 10000 == 0:  # Adjust frequency as needed
                        logger.info(f"  Processing state {state_idx+1}/{len(states)} in iteration {i+1}")
                        
                    if state_idx == 10000:
                        logger.info('Reached 10000 states, stopping early for demonstration purposes.')
                        break
                    
                    maxvalue = -100
                    
                    fen = self.make_fen(base_fen, s, "w")

                    game = Game()
                    game.reset_from_fen(fen)
                    enviroment = Env(game, gamma = self.gamma, step_penalty = self.step_penalty)
                    color = enviroment.state().get_side_to_move()

                    # Check if it's already checkmate
                    if (game.get_check(color)):
                        values[state_to_idx[s]]= 0
                        newPolicy[s] = None
                        continue

                    # it "tries out" all actions and store the best
                    for A in enviroment.state().legal_moves(color):
                        game.do_move(A)

                        # we assume that black is deterministic and makes always the best move
                        best_move_uci = self.get_black_move(game.to_fen())
                        if best_move_uci is None:
                            # If no move from tablebase, skip this action
                            game.undo_move(A)
                            continue
                        
                        best_move = Move.from_uci(game, best_move_uci)
                        game.do_move(best_move)

                        s_new = tuple(parse_fen_pieces(game.to_fen()))

                        R = 1 if game.get_check(color) else 0
                        
                        value_action = R + values[state_to_idx[s_new]]
                            
                        if (value_action > maxvalue):
                            maxvalue = value_action
                            bestact = A

                        game.undo_move(best_move)
                        game.undo_move(A)

                    # It stores the new value and policy of state S
                    newValues[state_to_idx[s]] = maxvalue
                    newPolicy[s] = bestact

            #Estimate change
            err = np.sqrt(np.mean( (newValues - values)**2))
            if err < self.tolerance:
                logger.debug('Distance between V_{}(S) and V_{}(S) is: {}'.format(i, i+1, err))
                break

        logger.info('Training completed.')
        logger.debug(f'Best policy found: {newPolicy}')

        # Save policy to file
        with open(self.save_path, "wb") as f:
            # Convert Move objects to UCI strings for pickling
            serializable_policy = {}
            for state, move in newPolicy.items():
                if move is not None:
                    serializable_policy[state] = f"{str(move.from_square)}{str(move.to_square)}"
                else:
                    serializable_policy[state] = None
            
            pickle.dump(serializable_policy, f)

        return newPolicy
