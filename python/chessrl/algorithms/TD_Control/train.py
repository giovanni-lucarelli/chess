#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import logging 
import os
from chessrl.utils.load_config import load_config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import argparse
from chessrl.utils.endgame_loader import load_positions
import numpy as np

# chess
from chessrl.algorithms.TD_Control.TD_Control import TD_Control

if __name__ == '__main__':

    ENDGAME_TYPE = config["endgame_type"]
    # Load positions
    positions, dtz_groups = load_positions(csv_path=f'../../../../tablebase/{ENDGAME_TYPE}/{ENDGAME_TYPE}_train.csv')
    endgames = [pos['fen'] for pos in positions]
    train_endgames = np.random.choice(endgames, size=config['n_episodes'], replace=False)
    logger.info(f'Loaded {len(train_endgames)} training positions for endgame {ENDGAME_TYPE}')

    for td_error_algorithm in ["QLearning", "SARSA"]:
        td_algo = TD_Control()
        td_algo.train(endgames=train_endgames, td_error_algorithm=td_error_algorithm)
    