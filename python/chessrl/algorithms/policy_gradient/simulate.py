#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import os
import glob
import logging
import argparse 
from chessrl.utils.load_config import load_config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt # type: ignore
from chessrl.utils.endgame_loader import load_positions
import numpy as np

# chess
from chessrl.utils.plot_chess import plot_game
from chessrl.algorithms.policy_gradient.reinforce import Policy, REINFORCE
from chessrl import chess_py as cp
from chessrl import Env, SyzygyDefender

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test REINFORCE policy on chess endgames')
    parser.add_argument('--weights', type=str, default='output/weights.pth',
                       help='Path to the model weights file. Default: output/weights.pth')
    parser.add_argument('--csv_path', type=str, default='../../../../tablebase/krk/krk_test.csv',
                       help='Path to the DTZ CSV file. Default: ../../../../tablebase/krk/krk_test.csv')
    parser.add_argument('--tb_path', type=str, default='../../../../tablebase/krk',
                       help='Path to Syzygy tablebase directory')
    
    args = parser.parse_args()
    
    reinforce = REINFORCE()
    reinforce.load_model(filepath=args.weights)
    
    positions, dtz_groups = load_positions(csv_path = args.csv_path)
    test_endgames = [pos['fen'] for pos in positions][:1000]

    endgames_won = 0
    testing_stats = []

    logger.info(f"Starting testing on {len(test_endgames)} endgames...")

    defender = SyzygyDefender(tb_path=args.tb_path)

    for endgame in test_endgames:
        env = Env.from_fen(
            endgame,
            defender=defender,
        )

        counter = 0
        while True:
            move = reinforce.policy.predict(env, reinforce.move_to_idx)
            step = env.step(move)
            counter += 1
            
            if counter > 50:
                logger.debug(f'!!! GAME STOPPED - Turn limit reached ({counter-1} turns) !!!')
                break
            
            if env.state().is_game_over():
                if step.reward > 0:
                    logger.debug('!!! CHECKMATE !!!')
                    endgames_won += 1
                else:
                    logger.debug('!!! GAME OVER (Draw/Stalemate) !!!')
                break
    
    
    logger.info("Testing completed.")
    logger.info(f"Endgames won: {endgames_won} / {len(test_endgames)} ({(endgames_won/len(test_endgames))*100:.2f}%)")
