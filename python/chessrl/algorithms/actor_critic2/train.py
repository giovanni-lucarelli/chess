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
    losses, rewards, all_ep_discounted = ac.train(
        train_endgames_1[:500],
        epochs=1500,
        device='cpu',
        max_steps=4
    )

    ac.save('output/actor_critic2_model_dtz1.pth', critic_path='output/actor_critic2_critic_dtz1.npy')

    # Saving losses, rewards, all_ep_discounted to csv
    csv_path = 'output/actor_critic2_dtz_1_results.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['loss', 'reward', 'episode_return_discounted'])
        for loss, reward, ep_return_discounted in zip(losses, rewards, all_ep_discounted):
            writer.writerow([loss, reward, ep_return_discounted])

    # DTZ = 3
    csv_path_train_3 = "../../../../tablebase/krk/krk_test.csv"   # <-- point to your CSV (fen,side,wdl,dtz)
    df_3 = pd.read_csv(csv_path_train_3)
    train_endgames_3 = df_3[df_3['dtz'] == 3]['fen'].tolist()

    # Start training DTZ=3
    losses, rewards, all_ep_discounted = ac.train(
        train_endgames_3[:500],
        epochs=1500,
        device='cpu',
        max_steps=8
    )

    ac.save('output/actor_critic2_model_dtz3.pth', critic_path='output/actor_critic2_critic_dtz3.npy')

    # Saving losses, rewards, all_ep_discounted to csv
    csv_path = 'output/actor_critic2_dtz_3_results.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['loss', 'reward', 'episode_return_discounted'])
        for loss, reward, ep_return_discounted in zip(losses, rewards, all_ep_discounted):
            writer.writerow([loss, reward, ep_return_discounted])