#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../../')

# utils
import logging
import os
import numpy as np
from chessrl.utils.load_config import load_config
from chessrl.utils.endgame_loader import get_all_endgames_from_dtz
import matplotlib.pyplot as plt

# Import the optimized REINFORCE
from chessrl.algorithms.reinforce.reinforce import Reinforce

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    reinforce = Reinforce(tb_path='../../../../tablebase/krk')
    
    logger.info(f"Model parameters: {sum(p.numel() for p in reinforce.policy.parameters()):,}")
        
    train_positions = get_all_endgames_from_dtz(csv_path=config['train_path'], dtz=1)
    train_endgames = [pos['fen'] for pos in train_positions]
    #train_endgames = np.random.choice(train_endgames, size=config['n_episodes'], replace=True).tolist()

    test_positions = get_all_endgames_from_dtz(csv_path=config['test_path'], dtz=1)
    test_endgames = [pos['fen'] for pos in test_positions]

    # Start training
    losses, rewards = reinforce.train(train_endgames, test_fens=test_endgames, epochs=config['epochs'], max_steps=config['max_steps'])
    
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