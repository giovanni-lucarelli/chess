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
from chessrl.algorithms.policy_gradient.policy import Policy
from chessrl.utils.move_idx import build_move_mappings

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
                 lr_v=0.01,
                 lr_a=0.001):
        """
        Calculates optimal policy using in-policy Temporal Difference control
        Approximate V-value for states S!
        """        
        # the discount factor
        self.gamma = gamma
        
        # the learning rate for value
        self.lr_v = lr_v
        # the learning rate for actor
        self.lr_a = lr_a
        
        # Stores the Value Approximation weights
        self.w = np.zeros((*self.space_size,))
        # Stores the Policy parametrization
        self.Theta = np.zeros( (*self.space_size, self.action_size) )

        self.defender = SyzygyDefender(tb_path=tb_path)

        self.policy = Policy()
    
    # -------------------   
    def single_step_update(self, s, a, r, new_s, done):
        """
        Uses a single step to update the values, using Temporal Difference delta for V values.
        """
        
        # ---------------------
        # CRITIC UPDATE -------
        # ---------------------
        
        if done:
            # -----------------------
            delta = (r + 0 - self.w[(*s,)])
            
        else:
            # --------------------------
            delta = (r + 
                      self.gamma * self.w[(*new_s,)]
                                 - self.w[(*s,)])
            
        # --------------------
        self.w[(*s,)] += self.lr_v * delta
        
        # -------------------------
        # Now Actor update --------
        # -------------------------
        policy = self.get_policy(s)
            
        for act in range(self.action_size):
            # If the action "act" is that which was really chosen in the trajectory
            if (a == act):
                self.Theta[(*s, act) ] += self.lr_a * delta * (1 - policy[act])
            # Else if the action "act" has not been performed
            else:
                self.Theta[(*s, act) ] += self.lr_a * delta * (- policy[act])

    def train(self, endgames):
        # Initialize learning rates
        # Note the difference in values!
        #Two time-scales!
        lr_v_0 = 0.025
        lr_a_0 = 0.001

        lr_v = lr_v_0
        lr_a = lr_a_0

        # Stochastically determined time-horizon!
        # Episode can end either by terminal state OR "killed" at each step with probability 1-gamma.
        gamma_discount = 1

        performance_traj_ActorCritic = np.zeros(len(endgames))

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
                a_idx = self.policy.get_action(env, legal_moves_idx)
                a = idx_to_move[a_idx] # returning UCI

                # Evolve one step
                step_result = env.step(a)

                new_s = env.state().to_fen()   
                r = step_result.reward      
                done = step_result.done

                new_x = self.obtain_features(new_s) 

                self.single_step_update([x],a,r,[new_x],done)    

                x = new_x  

                count += 1
