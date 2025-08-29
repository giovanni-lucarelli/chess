#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)

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

class ValueNN(nn.Module):
    def __init__(self, input_channels=12, filters=128, residual_blocks=6):
        super(ValueNN, self).__init__()
        
        # Same architecture as policy but different output
        self.initial_conv = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(filters)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(filters) for _ in range(residual_blocks)
        ])
        
        # Value head
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1) # Returns a single scalar value -> V(s)
    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 12, 8, 8)
            
        Outputs:
            value: Tensor of shape (batch_size,) representing the value of each state
        """
        # Initial convolution
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)
        
        return value.squeeze()