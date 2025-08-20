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
from value_iteration import ValueIteration 

if __name__ == '__main__':
    valueIteration = ValueIteration()
    
    # For now we are using only RRK vs K
    best_policy = valueIteration.train(config['endgames'][0])


