import streamlit as st
from PIL import Image, ImageDraw
import chessengine_py # type: ignore

# Constants for chessboard
ROWS, COLS = 8, 8
SQUARE_SIZE = 90  # pixels for each square
LIGHT_COLOR = (240, 217, 181)  # light square color
DARK_COLOR = (181, 136, 99)     # dark square color

# Load piece images
piece_image_paths = {
    "w0": "assets/w_pawn.svg",    
    "w1": "assets/w_knight.svg",  
    "w2": "assets/w_bishop.svg",  
    "w3": "assets/w_rook.svg",    
    "w4": "assets/w_queen.svg",   
    "w5": "assets/w_king.svg",    
    "b0": "assets/b_pawn.svg",    
    "b1": "assets/b_knight.svg",  
    "b2": "assets/b_bishop.svg",  
    "b3": "assets/b_rook.svg",    
    "b4": "assets/b_queen.svg",   
    "b5": "assets/b_king.svg"     
}

piece_images = {}
for piece, path in piece_image_paths.items():
    piece_images[piece] = Image.open(path).convert("RGBA")

def create_board_image(board, highlight_square=None, legal_moves=None):
    width = COLS * SQUARE_SIZE
    height = ROWS * SQUARE_SIZE
    board_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(board_img)
    
    # Draw the board and pieces.
    for disp_row, board_row in enumerate(reversed(board)):
        for col in range(COLS):
            is_light_square = ((7 - disp_row) + col) % 2 == 0
            color = LIGHT_COLOR if is_light_square else DARK_COLOR
            
            x1 = col * SQUARE_SIZE
            y1 = disp_row * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
            piece_code = board_row[col]
            if piece_code.strip():
                if piece_code in piece_images:
                    piece_img = piece_images[piece_code].rotate(180, expand=True)
                    board_img.paste(piece_img, (x1, y1), piece_img)
                else:
                    print(f"Warning: {piece_code} image not found.")
    
    # Highlight legal move squares with a blue border.
    if legal_moves is not None:
        for (row, col) in legal_moves:
            # Since the board is drawn in reversed order, adjust the row.
            disp_row = ROWS - 1 - row
            x1 = col * SQUARE_SIZE
            y1 = disp_row * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
    
    # Highlight the king in check with a red border.
    if highlight_square is not None:
        (row, col) = highlight_square
        disp_row = ROWS - 1 - row
        x1 = col * SQUARE_SIZE
        y1 = disp_row * SQUARE_SIZE
        x2 = x1 + SQUARE_SIZE
        y2 = y1 + SQUARE_SIZE
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    
    # Add labels on the bottom row (columns) and left column (rows).
    columns = "abcdefgh"
    
    for col in range(COLS):
        x = col * SQUARE_SIZE + SQUARE_SIZE/2 - 5 
        y = height - 12  
        draw.text((x, y), columns[col], fill="black")
    
    for row in range(ROWS):
        number = str(8 - row)
        x = 5 
        y = row * SQUARE_SIZE + SQUARE_SIZE/2 - 5  
        draw.text((x, y), number, fill="black")
    
    return board_img

# Function to handle chess game moves
def handle_move(game, from_square, to_square): 
    legal_moves = game.legal_moves(from_square) 
    for move in legal_moves: 
        if int(move.to) == int(to_square): 
            if game.is_move_legal(move): 
                game.do_move(move) 
                return True 
            return False

def init_state():
    if "game" not in st.session_state:
        st.session_state.game = chessengine_py.Game()
    if "turn" not in st.session_state:
        st.session_state.turn = 0

def main():
    init_state()
    game = st.session_state.game

    st.title("Chess Game")
    board_placeholder = st.empty()

    # Render and display the initial board only.
    board_state = game.get_board().get_board()
    img = create_board_image(board_state)
    board_placeholder.image(img, use_column_width=False)

    # Create a placeholder for displaying which side's turn it is.
    turn_placeholder = st.sidebar.empty()

    # Sidebar inputs for selecting piece and move
    side_to_move = game.get_side_to_move()
    if side_to_move == chessengine_py.Color.WHITE:
        turn_placeholder.write("White's turn")
    else:
        turn_placeholder.write("Black's turn")

    input_piece = st.sidebar.text_input("Enter piece to move (e.g., 'e2'):", "")
    legal_highlights = None
    if input_piece:
        try:
            from_square = chessengine_py.Square(
                ord(input_piece[0]) - ord('a') + 8 * (int(input_piece[1]) - 1)
            )
            moves = game.legal_moves(from_square)
            legal_moves_str = [chessengine_py.square_to_string(move.to) for move in moves]
            st.sidebar.write("Legal moves:", legal_moves_str)
            # Compute the board coordinate for each 'to' square.
            legal_highlights = []
            for move in moves:
                # Convert square number to (row, col):
                square_val = int(move.to)
                row = square_val // 8
                col = square_val % 8
                legal_highlights.append((row, col))
        except Exception as e:
            st.sidebar.error(f"Invalid input: {e}")
    
    input_move = st.sidebar.text_input("Enter move (e.g., 'e4'):", "")
    if st.sidebar.button("Make Move"):
        if input_piece and input_move:
            from_square = chessengine_py.Square(
                ord(input_piece[0]) - ord('a') + 8 * (int(input_piece[1]) - 1)
            )
            to_square = chessengine_py.Square(
                ord(input_move[0]) - ord('a') + 8 * (int(input_move[1]) - 1)
            )
            if handle_move(game, from_square, to_square):
                st.sidebar.success("Move successful!")
                st.session_state.turn += 1
                board_state = game.get_board().get_board()
                legal_highlights = None  # Clear legal move highlights after move.
                # Update the sidebar turn message.
                side_to_move = game.get_side_to_move()
                if side_to_move == chessengine_py.Color.WHITE:
                    turn_placeholder.write("White's turn")
                else:
                    turn_placeholder.write("Black's turn")
                img = create_board_image(board_state, legal_moves=legal_highlights)
                board_placeholder.image(img, caption="Current Board", use_column_width=False)
            else:
                st.sidebar.error("Illegal move!")
    
    st.sidebar.write(f"Turn: {st.session_state.turn}")
    game.check_control()

    # Check and print if a king is in check.
    if game.get_check(chessengine_py.Color.WHITE):
        st.sidebar.error("White King is in check!")
    if game.get_check(chessengine_py.Color.BLACK):
        st.sidebar.error("Black King is in check!")

    all_legal_moves = []
    for square_int in range(64):
        square = chessengine_py.Square(square_int)
        all_legal_moves.extend(game.legal_moves(square))

    if not all_legal_moves and game.get_check(game.get_side_to_move()):
        winner = "Black" if game.get_side_to_move() == chessengine_py.Color.WHITE else "White"
        st.sidebar.write(f"Checkmate! {winner} wins!")
        st.stop()  # Stop further execution

    # Get the updated board state.
    board_state = game.get_board().get_board()

    # Determine which king, if any, needs a red highlight.
    highlight_square = None
    if game.get_check(chessengine_py.Color.WHITE):
        for row_idx, row in enumerate(board_state):
            for col_idx, cell in enumerate(row):
                if cell == "w5":  # White king piece code
                    highlight_square = (row_idx, col_idx)
                    break
            if highlight_square is not None:
                break
    if game.get_check(chessengine_py.Color.BLACK):
        for row_idx, row in enumerate(board_state):
            for col_idx, cell in enumerate(row):
                if cell == "b5":  # Black king piece code
                    highlight_square = (row_idx, col_idx)
                    break
            if highlight_square is not None:
                break

    # Render the final board image with both blue highlights (legal moves) and, if applicable, a red highlight.
    img = create_board_image(board_state, highlight_square, legal_moves=legal_highlights)
    board_placeholder.image(img, caption="Current Board", use_column_width=False)

if __name__ == "__main__":
    main()