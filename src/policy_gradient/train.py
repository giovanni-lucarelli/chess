#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import logging
from utils.load_config import load_config
config = load_config()
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# chess
from build.chess_py import Game, Env
from utils.plot_chess import plot_game, plot_fen
from reinforce import policy, REINFORCE 

if __name__ == '__main__':
    game = Game()

    reinforce = REINFORCE()

    game.reset_from_fen(config['endgames'][0])
    env = Env(game, gamma = config['gamma'], step_penalty = config['step_penalty'])

    best_policy = reinforce.train(config['endgames'][0]) # TODO: train on multiple endgames


