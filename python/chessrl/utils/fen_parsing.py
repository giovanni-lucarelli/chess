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

def parse_fen(fen):
    """
    Since we consider simple endgames (no pawns, no castling) where 
    white always win or draws, we consider only piece placement for the NN.
    """
    parts = fen.split(' ')
    
    # Board state (8x8x12 piece planes)
    board_tensor = parse_piece_placement(parts[0])  # [8, 8, 12]
    
    return board_tensor

def parse_fen_cached(fen: str, fen_cache) -> torch.Tensor:
        """
        Parse FEN with caching to avoid repeated computation.
        """
        if fen not in fen_cache:
            if len(fen_cache) >= 300000:
                # Remove oldest entries if cache is full
                fen_cache.pop(next(iter(fen_cache)))
            
            parsed = parse_fen(fen).permute(2, 0, 1)
            fen_cache[fen] = parsed
        
        return fen_cache[fen]