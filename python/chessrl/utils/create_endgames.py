#!/usr/bin/env python3


import random
import json
import numpy as np
import itertools
from chessrl.chess_py import Game, Color

def parse_fen_pieces(fen):
    """
    Extract pieces from FEN string.
    Returns list of (piece, square) tuples.
    """
    board_part = fen.split()[0]
    pieces = []
    
    square = 0
    for rank_data in board_part.split('/'):
        for char in rank_data:
            if char.isdigit():
                square += int(char)  # Skip empty squares
            else:
                pieces.append((char, square))
                square += 1
    
    return pieces

def pieces_to_board_string(piece_positions):
    """
    Convert piece positions back to FEN board representation.
    piece_positions: list of (piece, square) tuples
    """
    board = [None] * 64  # 8x8 board
    
    # Place pieces on board
    for piece, square in piece_positions:
        board[square] = piece
    
    # Convert to FEN format
    fen_ranks = []
    for rank in range(8):
        rank_str = ""
        empty_count = 0
        
        for file in range(8):
            square = rank * 8 + file
            piece = board[square]
            
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += piece
        
        if empty_count > 0:
            rank_str += str(empty_count)
        
        fen_ranks.append(rank_str)
    
    return '/'.join(fen_ranks)

def get_square_coords(square):
    """Convert square index (0-63) to (rank, file) coordinates."""
    return square // 8, square % 8

def are_kings_adjacent(king1_square, king2_square):
    """Check if two kings are adjacent (illegal position)."""
    rank1, file1 = get_square_coords(king1_square)
    rank2, file2 = get_square_coords(king2_square)
    
    return abs(rank1 - rank2) <= 1 and abs(file1 - file2) <= 1

def is_valid_square_for_piece(piece, square):
    """Check if a piece can legally be placed on a square."""
    rank, file = get_square_coords(square)
    
    # Pawns cannot be on 1st or 8th rank
    if piece.lower() == 'p':
        return 1 <= rank <= 6
    
    return True

def is_position_legal(piece_positions):
    """
    Check if a position is legal:
    - Kings not adjacent
    - Pawns not on 1st/8th rank
    - No pieces on same square
    """
    squares_used = set()
    kings = []
    
    for piece, square in piece_positions:
        # Check for duplicate squares
        if square in squares_used:
            return False
        squares_used.add(square)
        
        # Check piece-specific rules
        if not is_valid_square_for_piece(piece, square):
            return False
        
        # Track kings
        if piece.lower() == 'k':
            kings.append(square)
    
    # Check kings not adjacent
    if len(kings) == 2:
        if are_kings_adjacent(kings[0], kings[1]):
            return False
    
    return True

def is_in_check(piece_positions, king_color):
    """
    Simple check detection (basic implementation).
    This is a simplified version - you might want to use your chess engine instead.
    """
    # Find the king
    king_piece = 'K' if king_color == 'white' else 'k'
    king_square = None
    
    for piece, square in piece_positions:
        if piece == king_piece:
            king_square = square
            break
    
    if king_square is None:
        return False
    
    # This is a very basic check - in practice, you'd want to use
    # your chess engine's is_check() method after creating the position
    return False

def randomize_endgame_position(base_fen, max_attempts=1000):
    """
    Create a random endgame position with the same material as base_fen.
    
    Args:
        base_fen: Base FEN string to extract material from
        max_attempts: Maximum attempts to generate a legal position
    
    Returns:
        New FEN string with randomized piece positions
    """
    # Parse the base FEN
    fen_parts = base_fen.split()
    original_pieces = parse_fen_pieces(base_fen)
    
    # Extract just the pieces (without positions)
    piece_types = [piece for piece, square in original_pieces]
    
    # Try to generate a legal position
    for attempt in range(max_attempts):
        # Generate random positions for all pieces
        available_squares = list(range(64))
        random.shuffle(available_squares)
        
        new_piece_positions = []
        for i, piece in enumerate(piece_types):
            square = available_squares[i]
            new_piece_positions.append((piece, square))
        
        # Check if position is legal
        if is_position_legal(new_piece_positions):
            # Create new FEN
            new_board = pieces_to_board_string(new_piece_positions)
            
            # Keep original game state info (active color, castling, etc.)
            new_fen = new_board + ' ' + ' '.join(fen_parts[1:])
            
            # Final validation using chess engine
            if validate_with_engine(new_fen):
                return new_fen
    
    # If we couldn't generate a legal position, return the original
    print(f"Warning: Could not generate legal position after {max_attempts} attempts")
    return base_fen

def validate_with_engine(fen):
    """
    Validate the position using the chess engine.
    Returns True if position is legal, False otherwise.
    """
    try:
        game = Game()
        game.reset_from_fen(fen)
        # If no exception was raised, the position is valid
        return True
    except:
        return False

def generate_endgame_positions(base_fen, num_positions):
    """
    Generate multiple randomized endgame positions.
    
    Args:
        base_fen: Base FEN to randomize from
        num_positions: Number of positions to generate
    
    Returns:
        List of FEN strings
    """
    positions = []
    generated = set()  # Avoid duplicates
    
    print(f"Generating {num_positions} endgame positions...")
    
    attempts = 0
    max_total_attempts = num_positions * 10  # Reasonable limit
    
    while len(positions) < num_positions and attempts < max_total_attempts:
        attempts += 1
        new_fen = randomize_endgame_position(base_fen)
        
        # Check for duplicates
        if new_fen not in generated:
            positions.append(new_fen)
            generated.add(new_fen)
            
            if len(positions) % 10 == 0:
                print(f"Generated {len(positions)}/{num_positions} positions...")
    
    if len(positions) < num_positions:
        print(f"Warning: Only generated {len(positions)} out of {num_positions} requested positions")
    
    return positions

def check_legality_with_engine(fen):
    game = Game()
    game.reset_from_fen(fen)
    game.set_side_to_move(Color.BLACK)
    # If black in check and still has moves
    if game.get_check(Color.BLACK) and any(game.legal_moves(Color.BLACK)):
        game.set_side_to_move(Color.WHITE)
        return False
    game.set_side_to_move(Color.WHITE)
    return True

def generate_all_endgame_positions(base_fen):
    # Parse the base FEN
    fen_parts = base_fen.split()
    original_pieces = parse_fen_pieces(base_fen)
    default_fen_part =' ' + ' '.join(fen_parts[1:])
    
    # Always keep both kings
    kings = [p for p in original_pieces if p[0].lower() == 'k']
    others = [p for p in original_pieces if p[0].lower() != 'k']
    
    positions = []
    state_to_idx = {}
    
    # Generate subsets of "others" (from size 0 up to all)
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            # Piece pool = 2 kings + subset
            piece_types = [piece for piece, _ in (kings + list(subset))]
            n = len(piece_types)

            # Generate all placements
            for piece_perm in set(itertools.permutations(piece_types)):
                for chosen_squares in itertools.combinations(range(64), n):
                    state = tuple(zip(piece_perm, chosen_squares))  # make hashable
                    fen = pieces_to_board_string(state) + default_fen_part 
                    if is_position_legal(state) and check_legality_with_engine(fen):
                            fen = pieces_to_board_string(state) + default_fen_part
                            if fen not in state_to_idx:  # avoid duplicates
                                idx = len(positions)
                                positions.append(fen)
                                state_to_idx[fen] = idx

    values = np.zeros(len(positions), dtype=float)
    
    print(f"Total states: {len(positions)}")

    print(positions[0])
    
    return positions, state_to_idx, values