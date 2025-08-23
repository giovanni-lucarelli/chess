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
from chessrl.utils.endgame_loader import sample_endgames

# chess
from chessrl.utils.plot_chess import plot_game
from chessrl.algorithms.policy_gradient.reinforce import Policy, REINFORCE
from chessrl import chess_py as cp
from chessrl import Env, SyzygyDefender

if __name__ == '__main__':
    # Clean previous plots
    plot_files = glob.glob('output/plots/turn_*.png')
    for file in plot_files:
        os.remove(file)
    logger.info(f'Cleaned {len(plot_files)} previous plot files')

    endgame = sample_endgames(csv_path='../../../../syzygy-tables/krk_dtz.csv', dtz_counts={1: 1})
    
    env = Env.from_fen(
        endgame[0]['fen'],
        step_penalty=0.01,
        defender=SyzygyDefender(tb_path='/Users/christianfaccio/UniTs/projects/chess/syzygy-tables'),   
    )
    reinforce = REINFORCE()
    reinforce.load_model(filepath='output/weights.pth')

    logger.info('Start simulating...')
    env.display_state(save_path="output/plots/turn_0.png")
    plt.show()
    counter = 1
    while True:
        move = reinforce.policy.predict(env.state(), reinforce.move_to_idx)
        step = env.step(move)
        env.display_state(save_path=f"output/plots/turn_{counter}.png")
        counter += 1
        
        if counter > 100:
            logger.info(f'!!! GAME STOPPED - Turn limit reached ({counter-1} turns) !!!')
            break
        
        if step.done:
            logger.info('!!! GAME OVER !!!')
            break
