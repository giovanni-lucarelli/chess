import streamlit as st
from PIL import Image, ImageDraw
import sys
sys.path.append("build")
import chessengine_py  # type: ignore

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
    
    # Add labels on the bottom row (columns) and left column (rows).
    columns = "abcdefgh"
    
    # Bottom labels: place letters in the center of each square at the bottom.
    for col in range(COLS):
        x = col * SQUARE_SIZE + SQUARE_SIZE/2 - 5  # adjust as needed
        y = height - 12  # a bit above the bottom edge
        draw.text((x, y), columns[col], fill="black")
    
    # Left labels: row numbers from 8 (top) to 1 (bottom)
    for row in range(ROWS):
        number = str(8 - row)
        x = 5  # a bit to the right of the left edge
        y = row * SQUARE_SIZE + SQUARE_SIZE/2 - 5  # center vertically
        draw.text((x, y), number, fill="black")
    
    return board_img

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

if __name__ == "__main__":
    main()