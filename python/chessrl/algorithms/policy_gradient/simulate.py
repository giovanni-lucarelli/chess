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
from chessrl.utils.endgame_loader import get_all_endgames_from_dtz, get_stats
import numpy as np

# chess
from chessrl.utils.plot_chess import plot_game
from chessrl.algorithms.policy_gradient.reinforce import Policy, REINFORCE
from chessrl import chess_py as cp
from chessrl import Env, SyzygyDefender

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test REINFORCE policy on chess endgames')
    parser.add_argument('--max_dtz', type=int, default=32, 
                       help='Maximum DTZ value for testing (range will be >0 and <max_dtz). Default: 32')
    parser.add_argument('--weights', type=str, default='output/weights.pth',
                       help='Path to the model weights file. Default: output/weights.pth')
    parser.add_argument('--csv_path', type=str, default='../../../../syzygy-tables/krk_dtz.csv',
                       help='Path to the DTZ CSV file. Default: ../../../../syzygy-tables/krk_dtz.csv')
    parser.add_argument('--tb_path', type=str, default='/Users/christianfaccio/UniTs/projects/chess/syzygy-tables',
                       help='Path to Syzygy tablebase directory')
    
    args = parser.parse_args()
    
    logger.info(f"Testing with max_dtz={args.max_dtz} (will test DTZ values >0 and <{args.max_dtz})")
    
    reinforce = REINFORCE()
    reinforce.load_model(filepath=args.weights)

    stats = get_stats(csv_path=args.csv_path)
    testing_stats = []

    # Filter DTZ values based on the max_dtz parameter
    available_dtz_values = [d for d in stats['dtz_distribution'] if d > 0 and d < args.max_dtz]
    
    if not available_dtz_values:
        logger.error(f"No DTZ values found in range (0, {args.max_dtz}). Available DTZ values: {sorted([d for d in stats['dtz_distribution'] if d > 0])}")
        sys.exit(1)
    
    logger.info(f"Found {len(available_dtz_values)} DTZ values to test: {sorted(available_dtz_values)}")

    for dtz in np.sort(available_dtz_values):
        logger.info(f"Testing on DTZ={dtz} with {stats['dtz_distribution'][dtz]} positions...")
        endgames = get_all_endgames_from_dtz(csv_path=args.csv_path, dtz=dtz)
        test_endgames = endgames[int(0.8 * len(endgames)):]  # last 20% for testing

        endgames_won = 0

        logger.info(f"Starting testing on {len(test_endgames)} endgames...")

        for endgame in test_endgames:
            env = Env.from_fen(
                endgame['fen'],
                step_penalty=0.01,
                defender=SyzygyDefender(tb_path=args.tb_path),
            )

            counter = 1
            while True:
                move = reinforce.policy.predict(env, reinforce.move_to_idx)
                step = env.step(move)
                counter += 1
                
                if counter > 50:
                    logger.debug(f'!!! GAME STOPPED - Turn limit reached ({counter-1} turns) !!!')
                    break
                
                if step.done:
                    if step.reward > 0:
                        logger.debug('!!! CHECKMATE !!!')
                        endgames_won += 1
                    else:
                        logger.debug('!!! GAME OVER (Draw/Stalemate) !!!')
                    break
        testing_stats.append((dtz, endgames_won, len(test_endgames)))
        logger.info(f"Testing completed for DTZ={dtz}. Won {endgames_won}/{len(test_endgames)} games ({(endgames_won/len(test_endgames))*100:.2f}%)")
    
    logger.info("Testing completed.")
    plt.figure(figsize=(10, 5))
    plt.plot([dtz for dtz, won, total in testing_stats], [won/total for dtz, won, total in testing_stats], marker='o')
    plt.title(f"Testing Results (DTZ range: 1 to {args.max_dtz-1})")
    plt.xlabel("DTZ")
    plt.ylabel("Win Rate")
    plt.grid()
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/testing_results.png")
    plt.show()
