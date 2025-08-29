#!/usr/bin/env python3 

# system
import sys 
sys.path.insert(0, '../../')

# utils
import logging
import os
import argparse
import numpy as np
import torch
from chessrl.utils.load_config import load_config
from chessrl.utils.endgame_loader import load_positions, get_stats, get_all_endgames_from_dtz
import matplotlib.pyplot as plt

# Import the optimized Actor-Critic with batch support
from chessrl.algorithms.actor_critic.ac_batch import ActorCritic

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":    
    # Load positions
    positions, dtz_groups = load_positions(csv_path=config['csv_path'])
    endgames = [pos['fen'] for pos in positions]
    train_endgames = np.random.choice(endgames, size=config['n_episodes'], replace=True).tolist()

    # Initialize Actor-Critic with batch training
    AC = ActorCritic() 
    
    # Train
    logger.info(f"Starting batch training with {config['n_episodes']} episodes...")
    losses, rewards = AC.train(train_endgames)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curve (now shows batch losses)
    if losses:
        ax1.plot(losses)
        ax1.set_xlabel('Batch Update')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Loss (Batch Size: 32)')
        ax1.grid(True)
    else:
        ax1.text(0.5, 0.5, 'No losses recorded', ha='center', va='center')
        ax1.set_title('Training Loss')

    # Rewards curve (per episode)
    if rewards:
        # Smooth the rewards curve for better visualization
        window_size = min(10, len(rewards) // 10) if len(rewards) > 10 else 1
        if window_size > 1:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(rewards, alpha=0.3, label='Raw')
            ax2.plot(range(window_size-1, len(rewards)), smoothed_rewards, label='Smoothed')
            ax2.legend()
        else:
            ax2.plot(rewards)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Reward per Episode')
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'No rewards recorded', ha='center', va='center')
        ax2.set_title('Average Reward per Episode')

    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/training.png')
    logger.info("Training curves saved to 'output/training.png'")