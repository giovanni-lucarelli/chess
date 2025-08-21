#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import os
import glob
import logging 
from utils.load_config import load_config
from utils.create_endgames import parse_fen_pieces
config = load_config()
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt # type: ignore

# chess
from utils.plot_chess import plot_game
from src.value_iteration.value_iteration import ValueIteration
from build.chess_py import Game, Env, Move
import chess
import chess.syzygy
import pickle

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

if __name__ == '__main__':
    # Clean previous plots
    plot_files = glob.glob('output/plots/turn_*.png')
    for file in plot_files:
        os.remove(file)
    logger.info(f'Cleaned {len(plot_files)} previous plot files')
    
    game = Game()
    # Load the policy dictionary from file
    with open(config['savepath_value_iteration'], "rb") as f:
        policy = pickle.load(f)
    game.reset_from_fen(config['test_endgames'][0])
    env = Env(game, gamma = config['gamma'], step_penalty = config['step_penalty'])
    
    logger.info('Start simulating...')
    plot_game(game, save_path=f'output/plots/turn_0.png', title='Initial Position')
    plt.show()
    counter = 1
    while True:
        current_fen = game.to_fen()
        pieces = tuple(parse_fen_pieces(current_fen))
        
        logger.info(f'Current FEN: {current_fen}')
        logger.info(f'Pieces tuple: {pieces}')
        
        # Check if this state exists in the policy
        if pieces not in policy:
            logger.error(f'State {pieces} not found in policy!')
            
        move_uci = policy[pieces] 
        if move_uci is None:
            logger.info('No valid white move found in policy, stopping simulation.')
            break
        logger.info(f'Policy UCI move: {move_uci}')
        
        move = Move.from_uci(game, move_uci)  # Convert to Move object
        game.do_move(move)

        if game.is_game_over():
            logger.info('!!! CHECKMATE !!!')
            break
        
        best_move_uci = get_black_move(game.to_fen())
        if best_move_uci is None:
            logger.info('No valid black move found, stopping simulation.')
            break
        
        best_move = Move.from_uci(game, best_move_uci)
        game.do_move(best_move)
        
        plot_game(game, save_path=f'output/plots/turn_{counter}.png', title=f'Turn {counter}')
        counter += 1
        
        if counter > 150:
            logger.info(f'!!! GAME STOPPED - Turn limit reached ({counter-1} turns) !!!')
            break
        
        if game.is_game_over():
            logger.info('!!! GAME OVER !!!')
            break
