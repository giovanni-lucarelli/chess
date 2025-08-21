#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import os
import glob
import logging 
import requests
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
        """Query online tablebase (Lichess API)"""
        url = f"http://tablebase.lichess.ovh/standard?fen={fen}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'moves' in data and data['moves']:
                # Get the best move (first in the list)
                best_move_data = data['moves'][0]
                return best_move_data['uci']  # Return UCI string directly
            return None
        except:
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
        pieces = tuple(parse_fen_pieces(game.to_fen()))
        move_uci = policy[pieces]  # This is now a UCI string
        if move_uci is None:
            logger.info('No valid move found in policy, stopping simulation.')
            break
        move = Move.from_uci(game, move_uci)  # Convert back to Move object
        game.do_move(move)
        game.do_move(move)
        best_move_uci = get_black_move(game.to_fen())
        best_move = chess.Move.from_uci(game, best_move_uci)
        game.do_move(best_move)
        
        plot_game(game, save_path=f'output/plots/turn_{counter}.png', title=f'Turn {counter}')
        counter += 1
        
        if counter > 150:
            logger.info(f'!!! GAME STOPPED - Turn limit reached ({counter-1} turns) !!!')
            break
        
        if game.is_game_over():
            logger.info('!!! GAME OVER !!!')
            break
