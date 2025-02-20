import streamlit as st
from PIL import Image, ImageDraw

# 1. Board constants
ROWS, COLS = 8, 8
SQUARE_SIZE = 90   # pixels for each square
LIGHT_COLOR = (240, 217, 181)  # light square color
DARK_COLOR  = (181, 136,  99)  # dark square color

# 2. Example board state
board_state = [
    ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
    ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
    ["",   "",   "",   "",   "",   "",   "",   ""  ],
    ["",   "",   "",   "",   "",   "",   "",   ""  ],
    ["",   "",   "",   "",   "",   "",   "",   ""  ],
    ["",   "",   "",   "",   "",   "",   "",   ""  ],
    ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
    ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
]

# 3. Piece image dictionary
piece_image_paths = {
    "wP": "assets/w_pawn.svg",
    "wR": "assets/w_rook.svg",
    "wN": "assets/w_knight.svg",
    "wB": "assets/w_bishop.svg",
    "wQ": "assets/w_queen.svg",
    "wK": "assets/w_king.svg",
    "bP": "assets/b_pawn.svg",
    "bR": "assets/b_rook.svg",
    "bN": "assets/b_knight.svg",
    "bB": "assets/b_bishop.svg",
    "bQ": "assets/b_queen.svg",
    "bK": "assets/b_king.svg"
}

# Preload piece images (optional but often faster)
piece_images = {}
for piece, path in piece_image_paths.items():
    piece_images[piece] = Image.open(path).convert("RGBA")

# 4. Function to create a single PIL image of the entire chessboard
def create_board_image(board):
    width  = COLS * SQUARE_SIZE
    height = ROWS * SQUARE_SIZE
    board_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(board_img)

    for row in range(ROWS):
        for col in range(COLS):
            # Determine square color
            is_light_square = (row + col) % 2 == 1
            color = LIGHT_COLOR if is_light_square else DARK_COLOR
            
            x1 = col * SQUARE_SIZE
            y1 = row * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            
            draw.rectangle([x1, y1, x2, y2], fill=color)

            piece_code = board[row][col]
            if piece_code:
                # Rotate each piece 180º before pasting
                piece_img = piece_images[piece_code].rotate(180, expand=True)
                board_img.paste(piece_img, (x1, y1), piece_img)

    return board_img

# 5. Streamlit UI
def main():
    # -- Place two input fields in the Streamlit sidebar --
    input_square_1 = st.sidebar.text_input("Select piece to move:", "")
    input_square_2 = st.sidebar.text_input("Insert square to move the piece:", "")
    
    # Display the values in the sidebar (optional)
    if input_square_1:
        st.sidebar.write("Select piece to move:", input_square_1)
    if input_square_2:
        st.sidebar.write("Insert square to move the piece:", input_square_2)
    
    # -- Display the board in the main area --
    img = create_board_image(board_state)
    st.image(img, caption="Current Board", use_column_width=False)

if __name__ == "__main__":
    main()
