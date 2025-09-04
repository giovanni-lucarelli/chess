#!/usr/bin/env python3

import sys
sys.path.insert(0, '../../../')

import os
import time
import matplotlib.pyplot as plt
from PIL import Image
from chessrl import Env, SyzygyDefender
from chessrl.utils.plot_chess import plot_fen

def create_value_iteration_gif(fen, move_fn, defender, output_dir="output/gif_frames", 
                              max_plies=100, gif_filename="value_iteration_game.gif"):
    """
    Create a series of images for a GIF showing Value Iteration gameplay for a single FEN.
    
    Args:
        fen (str): Starting FEN position
        move_fn: Function that takes a FEN and returns a UCI move
        defender: Defender object (e.g., SyzygyDefender)
        output_dir (str): Directory to save individual frame images
        max_plies (int): Maximum number of plies to play
        gif_filename (str): Name of the final GIF file
    
    Returns:
        dict: Game statistics including success, plies, and timing info
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    env = Env.from_fen(fen, defender=defender, absorb_black_reply=True)
    
    frame_paths = []
    move_count = 0
    total_time = 0.0
    
    # Save initial position
    initial_title = f"Initial Position\nFEN: {fen[:50]}..."
    initial_path = os.path.join(output_dir, f"frame_{move_count:03d}_initial.png")
    plot_fen(fen, title=initial_title, save_path=initial_path, size=800)
    frame_paths.append(initial_path)
    
    while env.steps() < max_plies and not env.is_terminal():
        move_count += 1
        root_fen = env.to_fen()
        
        # Get move from Value Iteration
        t0 = time.perf_counter()
        uci = move_fn(root_fen)
        move_time = time.perf_counter() - t0
        total_time += move_time
        
        if not uci:
            print(f"No move found for position: {root_fen}")
            break
            
        # Execute the move
        sr = env.step(uci)
        
        # Create title with move information
        title = f"Move {move_count}: {uci}\n"
        title += f"Time: {move_time*1000:.1f}ms\n"
        title += f"Plies: {env.steps()}"
        
        # Save frame after move
        frame_path = os.path.join(output_dir, f"frame_{move_count:03d}_after_{uci}.png")
        plot_fen(env.to_fen(), title=title, save_path=frame_path, size=800, lastmove=uci)
        frame_paths.append(frame_path)
        
        if sr.done:
            break
    
    # Create final frame with game result
    final_move_count = move_count + 1
    try:
        is_checkmate = env.state().is_checkmate()
        result_text = "Checkmate!" if is_checkmate else "Game Over"
        success = int(is_checkmate)
    except Exception:
        result_text = "Game Over"
        success = 0
    
    final_title = f"{result_text}\n"
    final_title += f"Total Moves: {move_count}\n"
    final_title += f"Total Plies: {env.steps()}\n"
    final_title += f"Avg Time/Move: {1000*total_time/max(move_count,1):.1f}ms"
    
    final_path = os.path.join(output_dir, f"frame_{final_move_count:03d}_final.png")
    plot_fen(env.to_fen(), title=final_title, save_path=final_path, size=800)
    frame_paths.append(final_path)
    
    # Create GIF
    create_gif_from_frames(frame_paths, os.path.join(output_dir, gif_filename))
    
    # Game statistics
    stats = {
        "fen": fen,
        "total_moves": move_count,
        "total_plies": env.steps(),
        "success": success,
        "total_time_s": total_time,
        "ms_per_move": 1000.0 * total_time / max(move_count, 1),
        "frame_count": len(frame_paths),
        "gif_path": os.path.join(output_dir, gif_filename)
    }
    
    print(f"Created GIF with {len(frame_paths)} frames: {stats['gif_path']}")
    print(f"Game result: {'Success' if success else 'Failed'} in {move_count} moves ({env.steps()} plies)")
    
    return stats

def create_gif_from_frames(frame_paths, gif_path, duration=1500, loop=0):
    """
    Create a GIF from a list of image file paths.
    
    Args:
        frame_paths (list): List of paths to frame images
        gif_path (str): Output path for the GIF
        duration (int): Duration per frame in milliseconds
        loop (int): Number of loops (0 = infinite)
    """
    if not frame_paths:
        print("No frames to create GIF")
        return
    
    # Load all images
    images = []
    for path in frame_paths:
        img = Image.open(path)
        # Convert to RGB if necessary (for GIF compatibility)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
    
    # Save as GIF
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    print(f"GIF saved: {gif_path}")

# Example usage
if __name__ == "__main__":
    # Example with a simple greedy policy (you'll need to adapt this to your actual Value Iteration policy)
    def dummy_move_fn(fen):
        """Placeholder move function - replace with your actual Value Iteration policy"""
        import chess
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        return str(legal_moves[0]) if legal_moves else None
    
    # Example FEN (KRK endgame)
    test_fen = "8/8/8/8/8/8/1k6/K6R w - - 0 1"
    
    # You'll need to replace this with your actual defender
    # defender = SyzygyDefender("../../../../tablebase/krk/")
    
    # For now, create a dummy defender for demonstration
    class DummyDefender:
        def __call__(self, fen):
            return "a1b1"  # dummy move
    
    defender = DummyDefender()
    
    stats = create_value_iteration_gif(
        fen=test_fen,
        move_fn=dummy_move_fn,
        defender=defender,
        output_dir="output/gif_frames",
        max_plies=50,
        gif_filename="value_iteration_demo.gif"
    )
    
    print("Game Statistics:", stats)