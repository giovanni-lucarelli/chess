#!/usr/bin/env python3 

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization as used in AlphaZero.
    
    Key features:
    - Skip connections for gradient flow
    - Batch normalization for training stability
    - Two 3x3 convolutions per block
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv-bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection and final activation
        out += residual
        out = F.relu(out)
        
        return out

class Policy(nn.Module):
    """
    Smaller version of AlphaZero policy optimized for chess endgames (≤7 pieces).
    
    Rationale:
    - Fewer pieces → simpler patterns → fewer parameters needed
    - Faster training and inference
    - Better for REINFORCE which needs many episodes
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
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Proper weight initialization for chess networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        return policy_logits
    
    def get_action_probs(self, fen, legal_moves=None, log=False):
        """
        Get action probabilities.
        Args:
            - fen: str -> FEN string representing the board state
            - legal_moves: list[int] -> List of legal move indices
            - log: bool -> Whether to return log probabilities
        """
        logits = self.forward(fen)
        
        mask = torch.full_like(logits, float('-inf'))
        for move_idx in legal_moves:
            mask[0, move_idx] = 0
        logits = logits + mask

        if log:
            return torch.log_softmax(logits, dim=-1)
        else:
            return torch.softmax(logits, dim=-1)
