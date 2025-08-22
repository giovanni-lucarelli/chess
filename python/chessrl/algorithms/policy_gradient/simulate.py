#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import os
import glob
import logging 
from utils.load_config import load_config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt # type: ignore

# chess
from utils.plot_chess import plot_game
from src.policy_gradient.reinforce import Policy, REINFORCE
from build.chess_py import Game, Env # type: ignore

if __name__ == '__main__':
    # Clean previous plots
    plot_files = glob.glob('output/plots/turn_*.png')
    for file in plot_files:
        os.remove(file)
    logger.info(f'Cleaned {len(plot_files)} previous plot files')
    
    game = Game()
    reinforce = REINFORCE()
    reinforce.load_model(config['filepath_test'])
    policy = reinforce.policy
    game.reset_from_fen(config['endgames_test'][0])
    env = Env(game, step_penalty = 0.01)
    
    logger.info('Start simulating...')
    plot_game(game, save_path=f'output/plots/turn_0.png', title='Initial Position')
    plt.show()
    counter = 1
    while True:
        move = policy.predict(env.state(), reinforce.move_to_idx)
        game.do_move(move)
        step = env.step(move)
        plot_game(game, save_path=f'output/plots/turn_{counter}.png', title=f'Turn {counter}')
        counter += 1
        
        if counter > 150:
            logger.info(f'!!! GAME STOPPED - Turn limit reached ({counter-1} turns) !!!')
            break
        
        if step.done:
            logger.info('!!! GAME OVER !!!')
            break
