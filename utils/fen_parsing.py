#!/usr/bin/env python3

import torch # type: ignore

def parse_piece_placement(placement):
    """Convert piece placement to 8x8x12 tensor"""
    piece_to_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    
    board = torch.zeros(8, 8, 12)
    ranks = placement.split('/')
    
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                file_idx += int(char)  # Skip empty squares
            else:
                piece_idx = piece_to_idx[char]
                board[rank_idx, file_idx, piece_idx] = 1
                file_idx += 1
    
    return board

def parse_castling(castling_str):
    """Convert castling rights to binary vector"""
    rights = torch.zeros(4)  # [K, Q, k, q]
    if 'K' in castling_str: rights[0] = 1
    if 'Q' in castling_str: rights[1] = 1  
    if 'k' in castling_str: rights[2] = 1
    if 'q' in castling_str: rights[3] = 1
    return rights

def parse_en_passant(ep_str):
    """Convert en passant square to one-hot encoding"""
    en_passant = torch.zeros(64)
    if ep_str != '-':
        file = ord(ep_str[0]) - ord('a')  # a=0, b=1, ...
        rank = int(ep_str[1]) - 1         # 1=0, 2=1, ...
        square_idx = rank * 8 + file
        en_passant[square_idx] = 1
    return en_passant

def parse_fen_to_features(fen_string):
    parts = fen_string.split(' ')
    
    # 1. Board state (8x8x12 piece planes)
    board_tensor = parse_piece_placement(parts[0])  # [8, 8, 12]
    
    # 2. Game state features
    active_color = 1 if parts[1] == 'w' else 0  # [1]
    castling_rights = parse_castling(parts[2])   # [4] (KQkq)
    en_passant = parse_en_passant(parts[3])      # [64] (one-hot square)
    halfmove_clock = int(parts[4]) / 50.0        # [1] (normalized)
    fullmove = int(parts[5]) / 100.0             # [1] (normalized)
    
    # Concatenate all features
    game_state = torch.cat([
        torch.tensor([active_color, halfmove_clock, fullmove]),
        castling_rights,
        en_passant
    ])  # Shape: [70]
    
    return board_tensor, game_state