#!/usr/bin/env python3

def build_move_mappings():
        """
        Build mappings between chess moves and action indices.
        Using custom square naming to match C++ implementation.
        move_to_idx -> a1:0 
        idx_to_move -> 0:a1
        """
        idx = 0
        move_to_idx = {}
        idx_to_move = {}
        
        # Generate square names manually (a1, b1, ..., h8)
        files = "abcdefgh"
        ranks = "12345678"
        squares = [f + r for r in ranks for f in files]
        
        # Generate all possible moves (from_square, to_square)
        for from_sq in squares:
            for to_sq in squares:
                if from_sq != to_sq:
                    move_str = from_sq + to_sq
                    move_to_idx[move_str] = idx
                    idx_to_move[idx] = move_str
                    idx += 1

        return move_to_idx, idx_to_move
    