#!/usr/bin/env python3 

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)

from chessrl.utils.fen_parsing import parse_fen

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization optimized for GPU.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)  # inplace for memory efficiency
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out, inplace=True)
        
        return out

class Policy(nn.Module):
    """
    AlphaZero-style policy network optimized for batch processing and GPU.
    """
    
    def __init__(self, 
                 input_channels=12,
                 filters=128,            
                 residual_blocks=6,      
                 policy_head_filters=16, 
                 action_size=4096):
        super(Policy, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(filters)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(filters) for _ in range(residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(filters, policy_head_filters, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_head_filters)
        self.policy_fc = nn.Linear(policy_head_filters * 8 * 8, action_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Batch of board states [batch_size, 12, 8, 8]
        
        Returns:
            Policy logits [batch_size, 4096]
        """
        # Initial convolution
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out, inplace=True)
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Policy head
        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy, inplace=True)
        policy = policy.reshape(policy.size(0), -1)  # Flatten for FC layer
        policy = self.policy_fc(policy)
        
        return policy
    
    def get_action(self, env, legal_moves_idx): 
        fen_tensor = parse_fen(env.to_fen()).unsqueeze(0).permute(0,3,1,2)  # [1, 8, 8, 12] -> [1, 12, 8, 8]
        logits = self.forward(fen_tensor) # action space [0, 4095]
        legal_logits = logits[0, legal_moves_idx]
        action_probs = torch.softmax(legal_logits, dim=-1)
        
        with torch.no_grad():
            action_idx = torch.multinomial(action_probs, 1).item()
            action = legal_moves_idx[action_idx]
        
        log_action_prob = torch.log(action_probs[action_idx]).squeeze()
        return action, log_action_prob # returns index in [0, 4095]
    
    def predict(self, state_tensor):
        """
        Predict the best move for a single state.
        
        Args:
            state_tensor: Single board state tensor [1, 12, 8, 8]
        
        Returns:
            Best move index (0-4095)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)
            best_move = torch.argmax(probs, dim=-1).item()
        return best_move # returns index in [0, 4095]