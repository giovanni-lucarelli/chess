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
from chessrl.utils.endgame_loader import get_all_endgames_from_dtz, get_stats

# chess
from chessrl.utils.plot_chess import plot_game, plot_fen
from chessrl.utils.create_endgames import generate_endgame_positions
from chessrl.algorithms.policy_gradient.reinforce import REINFORCE 

if __name__ == '__main__':
    # Initialize REINFORCE agent
    reinforce = REINFORCE()
    
    # Train the agent
    logger.info(f"Starting training for {config['n_episodes']} episodes...")

    endgames = get_all_endgames_from_dtz(csv_path='../../../../syzygy-tables/krk_dtz.csv', dtz=3) # training on KRvK with DTZ=3 only
    train_endgames = endgames[:int(0.8 * len(endgames))]

    reinforce.train(train_endgames)
    logger.info("Training completed successfully")

    # Save the trained model
    logger.info("Saving trained model...")
    reinforce.save_model()
    logger.info(f"Model saved to {config['filepath_train']}")
    
    # Plot training progress
    #logger.info("Plotting DTM progress...")
    #reinforce.plot_dtm_progress()


