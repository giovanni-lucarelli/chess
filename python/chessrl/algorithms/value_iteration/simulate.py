#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import os
import glob
import logging 
from chessrl.utils.load_config import load_config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt # type: ignore

# chess
from chessrl.env import Env, SyzygyDefender, LichessDefender
from chessrl.utils.io import load_policy_jsonl


if __name__ == '__main__':
    # Clean previous plots
    plot_files = glob.glob('output/plots/turn_*.png')
    for file in plot_files:
        os.remove(file)
    logger.info(f'Cleaned {len(plot_files)} previous plot files')

    # Load the policy dictionary from file
    policy = load_policy_jsonl("../../../../artifacts/policies/vi_krk_greedy.jsonl")


    TB_PATH = "tablebase"  
    TB_PATH = os.path.join(os.path.dirname(__file__), '..','..', '..', '..', 'tablebase', 'krk')
    #defender = SyzygyDefender(TB_PATH) 
    defender = LichessDefender()

    fens = config['test_endgames']

    logger.info('Start simulating with mode...')
    for fen in fens:
        env = Env.from_fen(fen, gamma = config['gamma'], defender = defender)
        logger.info("\n")

        counter = 1
        while True:
            current_fen = env.to_fen()

            logger.info(f'Current FEN: {current_fen}')

            # Check if this state exists in the policy
            if current_fen not in policy:
                logger.error(f'Current FEN not found in policy!')
                break

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
                if env.state().is_stalemate():
                    logger.info('Stalemate!')
                elif env.state().is_insufficient_material():
                    logger.info('Draw due to insufficient material!')
                else:
                    logger.info('Checkmate!')
                
                break