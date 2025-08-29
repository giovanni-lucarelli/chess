#!/usr/bin/env python3 

# system
import sys 
sys.path.insert(0, '../../')

# utils
import logging
import os
import argparse
import numpy as np
from chessrl.utils.load_config import load_config
from chessrl.utils.endgame_loader import load_positions, get_stats, get_all_endgames_from_dtz
import matplotlib.pyplot as plt

# Import the optimized REINFORCE
from chessrl.algorithms.actor_critic.ac import ActorCritic

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    positions, dtz_groups = load_positions(csv_path=config['csv_path'])
    endgames = [pos['fen'] for pos in positions]
    train_endgames = np.random.choice(endgames, size=config['n_episodes'], replace=True).tolist()

    AC = ActorCritic()
    losses, rewards = AC.train(train_endgames)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Loss curve
    ax1.plot(losses)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    # Rewards curve
    ax2.plot(rewards)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Average Reward per Batch')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('output/training.png')
    logger.info("Training curves saved to 'output/training.png'")