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
from chessrl.utils.plot_chess import plot_game, plot_fen
from chessrl.utils.create_endgames import generate_endgame_positions
from chessrl.algorithms.policy_gradient.reinforce import REINFORCE 

if __name__ == '__main__':
    logger.info("Starting REINFORCE training for chess endgames")
    
    # Initialize REINFORCE agent
    reinforce = REINFORCE()
    
    # Train the agent
    logger.info(f"Starting training for {config['n_episodes']} episodes...")
    #fen = "8/1k6/3R4/1K6/8/8/8/8 w - - 0 1" # KRvK
    #endgames = generate_endgame_positions(fen, config['n_endgames'])
    reinforce.train()
    logger.info("Training completed successfully")
    
    # Save the trained model
    logger.info("Saving trained model...")
    reinforce.save_model()
    logger.info(f"Model saved to {config['filepath_train']}")
    
    # Plot training progress
    #logger.info("Plotting DTM progress...")
    #reinforce.plot_dtm_progress()


