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
from build.chess_py import Env, Move
from utils.env_class import Env, SyzygyDefender

import pickle

if __name__ == '__main__':
    # Clean previous plots
    plot_files = glob.glob('output/plots/turn_*.png')
    for file in plot_files:
        os.remove(file)
    logger.info(f'Cleaned {len(plot_files)} previous plot files')
    
    # Load the policy dictionary from file
    with open(config['savepath_value_iteration'], "rb") as f:
        policy = pickle.load(f)
        
    TB_PATH = "tablebase"  
    defender = SyzygyDefender(TB_PATH) 
    env = Env.from_fen(config['test_endgames'][0], gamma = config['gamma'], step_penalty = config['step_penalty'], defender = defender)
    
    logger.info('Start simulating...')
    env.display_state()
    plt.show()
    counter = 1
    while True:
        current_fen = env.to_fen()
        
        logger.info(f'Current FEN: {current_fen}')
        
        # Check if this state exists in the policy
        if current_fen not in policy:
            logger.error(f'Current FEN not found in policy!')
            
        move_uci = policy[current_fen] 
        if move_uci is None:
            logger.info('No valid white move found in policy, stopping simulation.')
            break
        logger.info(f'Policy UCI move: {move_uci}')
        
        # Apply the move to the environment
        env.step(move_uci)
        
        counter += 1
        
        if counter > 150:
            logger.info(f'!!! GAME STOPPED - Turn limit reached ({counter-1} turns) !!!')
            break
        
        if env.state().is_game_over():
            logger.info('!!! GAME OVER !!!')
            break
