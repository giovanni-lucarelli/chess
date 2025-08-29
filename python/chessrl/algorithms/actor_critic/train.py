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
from chessrl.algorithms.actor_critic.ac import ActorCritic

if __name__ == "__main__":
    AC = ActorCritic()
    AC.train()