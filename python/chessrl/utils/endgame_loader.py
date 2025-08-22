import csv
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


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
                positions.append(fen)
        
        values = np.zeros(len(positions), dtype=float)

        return positions, positions_to_idx, values


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
    
def sample_by_single_dtz(dtz: int, count: int) -> List[Dict[str, Union[str, int]]]:
    """
    Convenience method to sample positions from a single DTZ value.
    
    Args:
        dtz: The DTZ value to sample from
        count: Number of positions to sample
        
    Returns:
        List of position dictionaries
    """
    return sample_endgames({dtz: count})
    
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
    stats = get_stats(csv_path='../../../syzygy-tables/krk_dtz.csv')
    print(f"Total positions: {stats['total_positions']}")
    print(f"Unique DTZ values: {stats['unique_dtz_values']}")
    print("DTZ distribution:", dict(sorted(stats['dtz_distribution'].items())))
    
    print("\nSampling 20 positions: 10 from DTZ=1, 5 from DTZ=3, 5 from DTZ=11")
    try:
        positions = sample_endgames(csv_path='../../../syzygy-tables/krk_dtz.csv', dtz_counts={1: 10, 3: 5, 11: 5})
        print(f"Successfully sampled {len(positions)} positions")
        
        # Show first few positions
        for i, pos in enumerate(positions[:3]):
            print(f"Position {i+1}: FEN={pos['fen']}, DTZ={pos['dtz']}")
            
    except ValueError as e:
        print(f"Error: {e}")