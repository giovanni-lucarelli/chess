#!/usr/bin/env python3

# system
import sys 
sys.path.insert(0, '../../../')

# utils
import numpy as np
import torch  
import logging
from tqdm import tqdm 
from chessrl.utils.load_config import load_config
from chessrl.utils.fen_parsing import parse_fen_cached
from typing import List, Tuple, Dict


import os
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(config_path)
logging.basicConfig(level=config['log_level'], format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# chess
from chessrl import Env, SyzygyDefender
from chessrl import chess_py as cp
from chessrl.algorithms.actor_critic.policy import Policy
from chessrl.utils.move_idx import build_move_mappings
from chessrl.utils.fen_parsing import parse_fen

move_to_idx, idx_to_move = build_move_mappings()

def get_legal_move_indices(env):
    """
    Get indices of legal moves for the current position.
    """
    legal_moves_idx = []
    
    for move in env.state().legal_moves(cp.Color.WHITE):
        move_str = cp.Move.to_uci(move)[:4]
        if move_str in move_to_idx:
            legal_moves_idx.append(move_to_idx[move_str])
    return legal_moves_idx

class ActorCritic():
    def __init__(self,  
                 tb_path: str = config['tb_path'],
                 gamma=1, 
                 lr_v=0.025,
                 lr_a=0.001):
        """
        Calculates optimal policy using in-policy Temporal Difference control
        Approximate V-value for states S!
        """        
        # the discount factor
        self.gamma = gamma
        
        # the learning rate for value
        self.lr_v = lr_v
        
        # Stores the Value Approximation weights
        self.w = np.zeros((*[9604],))

        self.defender = SyzygyDefender(tb_path=tb_path)

        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_a)


    
    def obtain_features(self, fen):
        """
        Extract features from FEN:
        - Distance of black king from nearest side of the board.
        - Horizontal & vertical distances of each white piece from black king
        (0 if adjacent).
        """
        board = parse_fen(fen)  # [8, 8, 12]
        
        # Find black king position
        bk_pos = torch.nonzero(board[:, :, 11], as_tuple=False)
        if bk_pos.size(0) == 0:
            raise ValueError("No black king found in FEN")
        bk_row, bk_col = bk_pos[0].tolist()
        
        # Distance of black king from side
        dist_side = min(bk_row, 7 - bk_row, bk_col, 7 - bk_col)
        
        features = [dist_side]
        
        # Loop over white pieces (indices 0..5)
        for piece_idx in range(6):
            positions = torch.nonzero(board[:, :, piece_idx], as_tuple=False)
            for r, c in positions.tolist():
                dx = max(0, abs(c - bk_col) - 1)
                dy = max(0, abs(r - bk_row) - 1)
                features.extend([dx, dy])
        
        return features

    
    # -------------------   
    def single_step_update(self, s, a_log_prob, r, new_s, done):
        """
        Uses a single step to update the values, using Temporal Difference delta for V values.
        """
        
        # ---------------------
        # CRITIC UPDATE -------
        # ---------------------
        
        if done:
            delta = (r + 0 - self.w[(*s,)])            
        else:
            delta = (r + self.gamma * self.w[(*new_s,)] - self.w[(*s,)])
            
        # --------------------
        self.w[(*s,)] += self.lr_v * delta
        
        # -------------------------
        # Now Actor update --------
        # -------------------------

        # REINFORCE loss
        delta_torch = torch.tensor(delta, dtype=torch.float32)
        loss = - (delta_torch * a_log_prob).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward() # calculates gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step() # updates weights

        return loss.item()

    def train(self, endgames):
        # Stochastically determined time-horizon!
        # Episode can end either by terminal state OR "killed" at each step with probability 1-gamma.
        gamma_discount = 1

        performance_traj_ActorCritic = np.zeros(len(endgames))

        losses = []
        rewards = []

        # RUN OVER EPISODES
        for i, s in enumerate(endgames):
            done = False
            x = self.obtain_features(s)
            count = 0
            while not done:

                env = Env.from_fen(
                    s,
                    defender = self.defender
                )
                
                # Legal moves idx for this state
                legal_moves_idx = get_legal_move_indices(env)

                # Select action from policy
                a_idx, a_log_prob = self.policy.get_action(env, legal_moves_idx)
                a = idx_to_move[a_idx] # returning UCI

                # Evolve one step
                step_result = env.step(a)

                new_s = env.state().to_fen()   
                r = step_result.reward      
                done = step_result.done

                new_x = self.obtain_features(new_s) 

                loss = self.single_step_update([x],a_log_prob,r,[new_x],done)    
                
                losses.append(loss)
                rewards.append(r)

                x = new_x  

                count += 1

        return losses, rewards
        

