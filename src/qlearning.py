#!/usr/bin/env python3

# os
import sys 
sys.insert(0, '../')

# utils
import logging 
from utils.load_config import load_config

config = load_config()
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# chess
from build.chess_py import Game,Env

class QLEARNING:
    """
    Class for the Q-learning algorithm.
    """
    def __init__(self):
        pass 
    def learn(self):
        pass 
    def predict(self):
        pass
