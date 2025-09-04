#!/usr/bin/env python3

# system
from random import random
import sys 
sys.path.insert(0, '../../../')

# utils
import logging
import os
import numpy as np
from chessrl.utils.load_config import load_config
from chessrl.utils.endgame_loader import get_all_endgames_from_dtz
import matplotlib.pyplot as plt
import torch
import csv
import pandas as pd

# Import the optimized REINFORCE
from chessrl.algorithms.actor_critic2.ac2 import ActorCritic

logging.basicConfig(level='INFO', format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    ac = ActorCritic(
        tb_path="../../../../tablebase/krk/",
        gamma=0.95,
        lr_v=2e-2,
        lr_a=1e-3,
        hidden=(128,128,64),
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in ac.policy.parameters()):,}")

    def load_fens_from_csv(csv_path, limit=None):
        fens = []
        with open(csv_path, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                fen = row["fen"]
                # make sure it's White to move
                if " w " in fen.split(" ", 2)[1:2] or fen.split()[1] == 'w':
                    fens.append(fen)
                    if limit and len(fens) >= limit: break
        return fens
    device = torch.device('cpu')

    csv_path_train_1 = "../../../../tablebase/krk/krk_train.csv"   # <-- point to your CSV (fen,side,wdl,dtz)
    df_1 = pd.read_csv(csv_path_train_1)
    train_endgames_1 = df_1[df_1['dtz'] == 1]['fen'].tolist()

    # Start training DTZ=1
    losses, rewards = ac.train(
        train_endgames_1[:500],
        epochs=1500,
        device='cpu',
        max_steps=4
    )

    ac.save('output/actor_critic2_model_dtz1.pth')

    # DTZ = 3
    csv_path_train_3 = "../../../../tablebase/krk/krk_test.csv"   # <-- point to your CSV (fen,side,wdl,dtz)
    df_3 = pd.read_csv(csv_path_train_3)
    train_endgames_3 = df_3[df_3['dtz'] == 3]['fen'].tolist()

    # Start training DTZ=3
    losses, rewards = ac.train(
        train_endgames_3[:500],
        epochs=1500,
        device='cpu',
        max_steps=8
    )

    ac.save('output/actor_critic2_model_dtz3.pth')
    """
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
    """