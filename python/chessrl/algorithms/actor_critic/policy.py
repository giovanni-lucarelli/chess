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
    AlphaZero-style policy network optimized for batch processing and GPU/MPS.
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
    
    def get_action(self, state, legal_moves_idx, predict=False): 
        """
        Get action from policy network, ensuring proper device placement.
        """
        if predict:
            self.eval()

        # Parse FEN and prepare tensor
        state_tensor = parse_fen(state).unsqueeze(0).permute(0,3,1,2)  # [1, 8, 8, 12] -> [1, 12, 8, 8]
        
        # Move to same device as model
        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)

        # Get logits, mask them for legal moves, sample an action based on those probs
        # (or select the one with highest prob if in prediction)
        logits = self.forward(state_tensor) # action space [0, 4095]
        logits = logits.squeeze(0) # [4096]
        logits_actions_dict = {k: logits[k] for k in range(4096)} # dict <move_idx, <logit_value>> -> {0: tensor(0.0245, grad_fn=<SelectBackward0>),...}
        legal_logits = {k: logits_actions_dict[k] for k in legal_moves_idx} # select only legal move' logits -> {2: tensor(0.7702, grad_fn=<SelectBackward0>),...}
        idxs = list(legal_logits.keys()) # list of legal move indices <move_idx> -> [2,3,4]
        values = torch.stack([legal_logits[k] for k in idxs]) # stack legal logits <logit_value> -> tensor([ 0.7702, -0.3072,  0.0914], grad_fn=<StackBackward0>)
        legal_probs = torch.softmax(values, dim=0) # obtain legal move probabilities -> tensor([0.5412, 0.1843, 0.2745], grad_fn=<SoftmaxBackward0>)
        legal_probs_dict = {k: v for k, v in zip(idxs, legal_probs)} # dict <move_idx, <probability>> -> {2: tensor(0.5412, grad_fn=<UnbindBackward0>),...}
        
        with torch.no_grad():
            if predict:
                action_idx = max(legal_probs_dict, key=legal_probs_dict.get) # selects best action on reduced action space (only legal moves) -> 2
            else:
                selected_idx = torch.multinomial(legal_probs, 1).item() # samples action on reduced action space (only legal moves) -> idx
                action_idx = idxs[selected_idx] # -> 2
        
        # Keep log prob for gradient computation - this needs gradients enabled
        log_legal_prob = torch.log(legal_probs_dict[action_idx]) # -> tensor(-0.6139, grad_fn=<LogBackward0>)
        
        return action_idx, log_legal_prob # returns index in [0, 4095] and relative log probability -> 2, tensor(-0.6139, grad_fn=<LogBackward0>)
    
    def batch_get_actions(self, envs, legal_moves_list):
        """
        Get actions for a batch of environments efficiently.
        
        Args:
            envs: List of chess environments
            legal_moves_list: List of legal move indices for each environment
        
        Returns:
            actions: List of selected action indices
            log_probs: Tensor of log probabilities for selected actions
        """
        batch_size = len(envs)
        if batch_size == 0:
            return [], torch.tensor([])
        
        # Parse all FENs and create batch tensor
        fen_tensors = []
        for env in envs:
            fen_tensor = parse_fen(env.to_fen()).permute(2, 0, 1)  # [12, 8, 8]
            fen_tensors.append(fen_tensor)
        
        batch_tensors = torch.stack(fen_tensors).to(next(self.parameters()).device, non_blocking=True)  # [batch_size, 12, 8, 8]
        
        # Single forward pass for entire batch
        with torch.no_grad():
            logits = self.forward(batch_tensors)  # [batch_size, 4096]
        
        # Process each environment's legal moves
        actions = []
        log_probs = []
        
        for i, (legal_moves_idx, env_logits) in enumerate(zip(legal_moves_list, logits)):
            if len(legal_moves_idx) == 0:
                # No legal moves (shouldn't happen in normal chess)
                actions.append(0)
                log_probs.append(torch.tensor(-float('inf')))
                continue
                
            legal_logits = env_logits[legal_moves_idx]
            
            # Check for NaN/inf values
            if torch.isnan(legal_logits).any() or torch.isinf(legal_logits).any():
                # Fallback to uniform distribution
                action_probs = torch.ones_like(legal_logits) / len(legal_logits)
            else:
                # Apply numerical stability
                legal_logits = torch.clamp(legal_logits, min=-50, max=50)
                legal_logits = legal_logits - torch.max(legal_logits)
                action_probs = torch.softmax(legal_logits, dim=-1)
                
                # Add epsilon and renormalize
                action_probs = action_probs + 1e-8
                action_probs = action_probs / torch.sum(action_probs)
            
            # Sample action
            action_idx = torch.multinomial(action_probs, 1).item()
            action = legal_moves_idx[action_idx]
            log_prob = torch.log(action_probs[action_idx])
            
            actions.append(action)
            log_probs.append(log_prob)
        
        return actions, torch.stack(log_probs) if log_probs else torch.tensor([])