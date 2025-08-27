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
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from chessrl.utils.endgame_loader import load_positions, get_stats, get_all_endgames_from_dtz

# chess
from chessrl.algorithms.policy_gradient.reinforce import REINFORCE 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train REINFORCE on chess endgames')
    parser.add_argument('--training_method', type=str, choices=['random', 'curriculum'], default='random',
                        help='Training method: "random" for random sampling, "curriculum" for DTZ-based curriculum learning. Default: random')
    parser.add_argument('--max_dtz', type=int, default=32, 
                        help='Maximum DTZ value for curriculum learning (range will be >0 and <max_dtz). Default: 32')
    parser.add_argument('--csv_path', type=str, default='../../../../tablebase/krk/krk_train.csv',
                       help='Path to the training CSV file. Default: ../../../../tablebase/krk/krk_train.csv')
    args = parser.parse_args()
    
    reinforce = REINFORCE() # initialize REINFORCE agent
    
    if args.training_method == 'curriculum':
        logger.info("Using DTZ-based curriculum learning method")
        
        # Get statistics about the dataset
        stats = get_stats(csv_path=args.csv_path)
        available_dtz_values = np.sort([d for d in stats['dtz_distribution'] if d > 0 and d < args.max_dtz]) 
        
        # !!! SORTING AND NOT SAMPLING IN REINFORCE MEANS USING A BASIC FORM OF CURRICULUM LEARNING !!!
        train_endgames = []
        weights = {}
        
        # Define distribution over DTZ of how many samples to use given the total episodes
        for dtz in available_dtz_values:
            weights[dtz] = np.log(dtz + 1) 

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()} 

        # Sample endgames according to DTZ distribution and sort by DTZ for curriculum learning
        for dtz in available_dtz_values:
            n_samples = int(normalized_weights[dtz] * config['n_episodes'])
            endgames = get_all_endgames_from_dtz(csv_path=args.csv_path, dtz=dtz)
            endgame_fens = [pos['fen'] for pos in endgames]
            if len(endgame_fens) > 0:
                samples = np.random.choice(endgame_fens, size=n_samples, replace=True)
                train_endgames.extend(samples)
        
        logger.info(f"Starting training for {config['n_episodes']} episodes using DTZ distribution:")
        weight_sum = 0
        for dtz, weight in normalized_weights.items():
            weight_sum += weight
            logger.info(f"DTZ {dtz}: {weight_sum:.4f} - Episodes: {int(weight_sum * config['n_episodes'])}")
        logger.info(f"Sum: {sum(normalized_weights.values()):.4f}")
        
    else:  # random sampling method
        logger.info("Using random sampling method")
        
        # Load all positions and sample randomly without considering DTZ
        positions, dtz_groups = load_positions(csv_path=args.csv_path)
        endgames = [pos['fen'] for pos in positions]
        train_endgames = np.random.choice(endgames, size=config['n_episodes'], replace=True)
    
    logger.info(f"Total training samples prepared: {len(train_endgames)}")
    reinforce.train(train_endgames) # TRAINING
    logger.info("Training completed successfully")



