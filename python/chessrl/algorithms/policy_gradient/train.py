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

# Import the optimized REINFORCE
from chessrl.algorithms.policy_gradient.reinforce import REINFORCE

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Optimized REINFORCE with GPU support')
    parser.add_argument('--training_method', type=str, choices=['random', 'curriculum'], default='random',
                        help='Training method: "random" for random sampling, "curriculum" for DTZ-based curriculum learning')
    parser.add_argument('--max_dtz', type=int, default=32, 
                        help='Maximum DTZ value for curriculum learning')
    args = parser.parse_args()
    
    reinforce = REINFORCE()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in reinforce.policy.parameters()):,}")
    
    # Load training data
    if args.training_method == 'curriculum':
        stats = get_stats(csv_path=config['csv_path'])
        available_dtz_values = np.sort([d for d in stats['dtz_distribution'] if d > 0 and d < args.max_dtz]) 
        
        train_endgames = []
        weights = {}
        
        for dtz in available_dtz_values:
            weights[dtz] = np.log(dtz + 1) 

        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()} 

        for dtz in available_dtz_values:
            n_samples = int(normalized_weights[dtz] * config['n_episodes'])
            endgames = get_all_endgames_from_dtz(csv_path=config['csv_path'], dtz=dtz)
            endgame_fens = [pos['fen'] for pos in endgames]
            if len(endgame_fens) > 0:
                samples = np.random.choice(endgame_fens, size=n_samples, replace=True)
                train_endgames.extend(samples)
        
    else:  # random sampling
        positions, dtz_groups = load_positions(csv_path=config['csv_path'])
        endgames = [pos['fen'] for pos in positions]
        train_endgames = np.random.choice(endgames, size=config['n_episodes'], replace=True).tolist()

    
    # Start training
    losses, rewards, dtms = reinforce.train(train_endgames)
    
    valid_dtms = [d for d in dtms if d != float('inf')]
    
    # Save training curves
    import matplotlib.pyplot as plt
    
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