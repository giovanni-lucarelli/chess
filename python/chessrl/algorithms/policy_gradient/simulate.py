#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../')

# utils
import os
import logging
import argparse 
from chessrl.utils.load_config import load_config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt # type: ignore
from chessrl.utils.endgame_loader import load_positions
from chessrl.utils.fen_parsing import parse_fen_cached
from chessrl.utils.move_idx import build_move_mappings
import torch

# chess
from chessrl.algorithms.policy_gradient.reinforce import REINFORCE
from chessrl import Env, SyzygyDefender

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test REINFORCE policy on chess endgames')
    parser.add_argument('--weights', type=str, default='output/weights.pth',
                       help='Path to the model weights file. Default: output/weights.pth')
    parser.add_argument('--csv_path', type=str, default='../../../../tablebase/krk/krk_test.csv',
                       help='Path to the DTZ CSV file. Default: ../../../../tablebase/krk/krk_test.csv')
    parser.add_argument('--tb_path', type=str, default='../../../../tablebase/krk',
                       help='Path to Syzygy tablebase directory')
    
    args = parser.parse_args()
    
    reinforce = REINFORCE()
    checkpoint = torch.load(args.weights)
    reinforce.policy.load_state_dict(checkpoint['model_state_dict'])
    
    # Build move mappings for converting indices to moves
    move_to_idx, idx_to_move = build_move_mappings()
    fen_cache = {}
    
    positions, dtz_groups = load_positions(csv_path = args.csv_path)
    # obtain only the FENs with DTZ=1
    test_endgames = [pos['fen'] for pos in positions if pos['dtz'] == 1]

    endgames_won = 0
    testing_stats = []

    logger.info(f"Starting testing on {len(test_endgames)} endgames...")

    defender = SyzygyDefender(tb_path=args.tb_path)

    for endgame in test_endgames:
        env = Env.from_fen(
            endgame,
            defender=defender,
        )

        counter = 0
        while True:
            # Get legal moves for current position
            from chessrl import chess_py as cp
            legal_move_indices = []
            for move in env.state().legal_moves(cp.Color.WHITE):
                move_str = cp.Move.to_uci(move)[:4]
                if move_str in move_to_idx:
                    legal_move_indices.append(move_to_idx[move_str])
            
            if not legal_move_indices:
                logger.debug('No legal moves available')
                break
            
            # Convert environment to tensor for prediction
            current_fen = env.to_fen()
            state_tensor = parse_fen_cached(current_fen, fen_cache).unsqueeze(0).to(reinforce.device)
            
            # Get policy logits and apply legal move mask
            with torch.no_grad():
                logits = reinforce.policy.forward(state_tensor)[0]  # Remove batch dimension
                
                # Create mask for legal moves
                mask = torch.zeros(4096, device=reinforce.device)
                mask[legal_move_indices] = 1
                
                # Apply mask and get probabilities
                masked_logits = logits.masked_fill(mask == 0, float('-inf'))
                action_probs = torch.softmax(masked_logits, dim=-1)
                
                # Select best legal move
                best_move_idx = torch.argmax(action_probs).item()
            
            move_str = idx_to_move[best_move_idx]
            move = cp.Move.from_strings(env.state(), move_str[:2], move_str[2:4])
            
            step = env.step(move)
            counter += 1
            
            if counter > 50:
                logger.debug(f'!!! GAME STOPPED - Turn limit reached ({counter-1} turns) !!!')
                break
            
            if env.state().is_game_over():
                if step.reward > 0:
                    logger.debug('!!! CHECKMATE !!!')
                    endgames_won += 1
                else:
                    logger.debug('!!! GAME OVER (Draw/Stalemate) !!!')
                break
    
    
    logger.info("Testing completed.")
    logger.info(f"Endgames won: {endgames_won} / {len(test_endgames)} ({(endgames_won/len(test_endgames))*100:.2f}%)")
