#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import os
import glob
import logging 
from chessrl.utils.load_config import load_config
config = load_config()
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt # type: ignore

# chess
from src.value_iteration.value_iteration import ValueIteration
from build.chess_py import Env, Move
from chessrl.env import Env, SyzygyDefender

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
    
    logger.info('Start simulating...')
    for fen in config['test_endgames']:
        env = Env.from_fen(fen, gamma = config['gamma'], step_penalty = config['step_penalty'], defender = defender)
        
        env.display_state()
        #plt.show()
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
            
            if counter > 50:
                logger.info(f'!!! GAME STOPPED - Turn limit reached ({counter-1} turns) !!!')
                break
            
            if env.state().is_game_over():
                logger.info('Checkmate!')
                break
