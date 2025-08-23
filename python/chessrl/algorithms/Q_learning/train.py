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

# chess
from Q_learning import QLearning_TDControl

if __name__ == '__main__':
    QLearning = QLearning_TDControl(config["endgame_type"], config["n_episodes"], config['max_steps'], config["gamma"], config["lr"], config["epsilon"], config["savepath_q_learning"])
    QLearning.train()
    optimal_policy = QLearning.save_greedy_policy()
    

