import csv
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from chessrl.env import Env
from chessrl import chess_py  # Import the chess_py module to access Move class


"""
Utility functions for loading and sampling chess endgame positions from CSV files.
Supports filtering by DTZ (Distance to Zeroing) values and sampling without replacement.
"""

def load_positions(csv_path: str):
    """
    Load chess endgame positions from a CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        A tuple containing:
        - A list of position dictionaries
        - A dictionary mapping DTZ values to lists of positions
    """
    positions = []
    dtz_groups = {}
    
    # Read the CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row['fen']
            wdl = int(row['wdl'])
            dtz = int(row['dtz'])
            pos = {'fen': fen, 'wdl': wdl, 'dtz': dtz}
            positions.append(pos)
            
            if dtz not in dtz_groups:
                dtz_groups[dtz] = []
            dtz_groups[dtz].append(pos)
    
    return positions, dtz_groups


def load_all_positions(csv_path: str):
        positions = []
        positions_to_idx = {}
        
        # Read the CSV file
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fen = row['fen']
                if ('b' in fen):
                    game = Env.from_fen(fen)
                    terminal_state = game.state().is_game_over()
                if ('w' in fen) or ('b' in fen and terminal_state):
                    positions_to_idx[fen] = len(positions)
                    positions.append(fen)
                else:
                    continue
            
            for fen in generate_two_kings_fens():
                if fen not in positions_to_idx:
                    positions_to_idx[fen] = len(positions)
                    positions.append(fen)
        
        values = np.zeros(len(positions), dtype=float)

        return positions, positions_to_idx, values

# TODO: this function is very ugly, make it nicer!! (Or I could delete it and make the csv contain these fens)
def generate_two_kings_fens():
    files = "abcdefgh"
    ranks = "12345678"

    def square(idx):
        return files[idx % 8] + ranks[idx // 8]

    def fen_row(pieces):
        """Convert list of 64 chars to FEN board string"""
        rows = []
        for r in range(8):
            row = pieces[r*8:(r+1)*8]
            fen_row = ""
            empty = 0
            for ch in row:
                if ch == ".":
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += ch
            if empty > 0:
                fen_row += str(empty)
            rows.append(fen_row)
        return "/".join(rows[::-1])  # ranks 8→1

    # Precompute adjacency (king moves)
    king_moves = {}
    for i in range(64):
        adj = []
        r, f = divmod(i, 8)
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    adj.append(nr*8 + nf)
        king_moves[i] = set(adj)

    fens = []
    for wk in range(64):
        for bk in range(64):
            if wk == bk:
                continue
            if bk in king_moves[wk]:  # kings adjacent → illegal
                continue
            # create empty board
            board = ["."] * 64
            board[wk] = "K"
            board[bk] = "k"
            fen_board = fen_row(board)
            for stm in ["w", "b"]:
                fen = f"{fen_board} {stm} - - 0 1"
                fens.append(fen)
    return fens

def load_all_positions_with_actions(csv_path: str):
        positions = []
        positions_actions_to_idx = {}
        # IMPORTANT DIFFERENCE: I don't save terminal states anymore!
        q_values = []
        last_index = 0
        
        # Read the CSV file
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fen = row['fen']
                game = Env.from_fen(fen)
                terminal_state = game.state().is_game_over()
                if ('w' in fen) and (not terminal_state):
                    legal_moves = list(game.state().legal_moves(game.state().get_side_to_move()))
                    for i, a in enumerate(legal_moves):
                        positions_actions_to_idx[(fen, chess_py.Move.to_uci(a))] = last_index+i
                    last_index += len(legal_moves)
                    positions.append(fen)
                else:
                    continue
            
        q_values = np.zeros(len(positions_actions_to_idx), dtype=float)             
        return positions, positions_actions_to_idx, q_values



def sample_endgames(
    csv_path: str,
    dtz_counts: Dict[int, int],
    total_limit: Optional[int] = None,
) -> List[Dict[str, Union[str, int]]]:
    """
    Sample endgame positions by DTZ values without replacement.
    
    Args:
        dtz_counts: Dictionary mapping DTZ values to number of samples needed
                    Example: {1: 10, 3: 5, 10: 2}
        total_limit: Optional maximum total number of positions to return
                    If specified and less than sum of dtz_counts, positions
                    will be sampled proportionally
                    
    Returns:
        List of position dictionaries with keys: 'fen', 'wdl', 'dtz'
    """

    positions, dtz_groups = load_positions(csv_path=csv_path)

    # Validate requests
    for dtz, count in dtz_counts.items():
        if dtz not in dtz_groups:
            raise ValueError(f"DTZ value {dtz} not found in dataset")
        if count > len(dtz_groups[dtz]):
            available = len(dtz_groups[dtz])
            raise ValueError(
                f"Requested {count} positions for DTZ={dtz}, "
                f"but only {available} available"
            )
    
    # Apply total limit if specified
    total_requested = sum(dtz_counts.values())
    if total_limit is not None and total_limit < total_requested:
        # Scale down proportionally
        scale_factor = total_limit / total_requested
        dtz_counts = {
            dtz: max(1, int(count * scale_factor))
            for dtz, count in dtz_counts.items()
        }
        
        # Adjust to exact total if needed
        current_total = sum(dtz_counts.values())
        if current_total != total_limit:
            # Add/remove from the DTZ with most positions
            dtz_with_most = max(dtz_counts.keys(), 
                                key=lambda x: len(dtz_groups[x]))
            dtz_counts[dtz_with_most] += (total_limit - current_total)
    
    # Sample positions
    sampled_positions = []
    for dtz, count in dtz_counts.items():
        if count > 0:
            positions = random.sample(dtz_groups[dtz], count)
            sampled_positions.extend(positions)
    
    # Shuffle the final result to mix DTZ groups
    random.shuffle(sampled_positions)
    
    return sampled_positions
    
def get_all_endgames_from_dtz(csv_path: str,
                              dtz: int):
    """
    Get all endgame positions for a specific DTZ value.
    """
    positions, dtz_groups = load_positions(csv_path=csv_path)
    return dtz_groups.get(dtz, [])

def get_stats(csv_path: str) -> Dict[str, Union[int, Dict[int, int]]]:
    """
    Get statistics about the loaded dataset.
    
    Returns:
        Dictionary with total count and DTZ distribution
    """
    positions, dtz_groups = load_positions(csv_path=csv_path)

    dtz_distribution = {dtz: len(positions)
                        for dtz, positions in dtz_groups.items()}

    return {
        'total_positions': len(positions),
        'unique_dtz_values': len(dtz_groups),
        'dtz_distribution': dtz_distribution
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Dataset statistics:")
    stats = get_stats(csv_path='../../../tablebase/krk/krk_test.csv')
    print(f"Total positions: {stats['total_positions']}")
    print(f"Unique DTZ values: {stats['unique_dtz_values']}")
    print("DTZ distribution:", dict(sorted(stats['dtz_distribution'].items())))
    
    print("\nSampling 20 positions: 10 from DTZ=1, 5 from DTZ=3, 5 from DTZ=11")
    try:
        positions = sample_endgames(csv_path='../../../tablebase/krk/krk_test.csv', dtz_counts={1: 10, 3: 5, 11: 5})
        print(f"Successfully sampled {len(positions)} positions")
        
        # Show first few positions
        for i, pos in enumerate(positions[:3]):
            print(f"Position {i+1}: FEN={pos['fen']}, DTZ={pos['dtz']}")
            
    except ValueError as e:
        print(f"Error: {e}")