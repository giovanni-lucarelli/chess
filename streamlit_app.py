import streamlit as st
from PIL import Image, ImageDraw
import sys
sys.path.append("build")
import chessengine_py  # type: ignore # Import your chess engine module

# Constants for chessboard
ROWS, COLS = 8, 8
SQUARE_SIZE = 90  # pixels for each square
LIGHT_COLOR = (240, 217, 181)  # light square color
DARK_COLOR = (181, 136, 99)  # dark square color

board_state = [
    ["b3", "b1", "b2", "b4", "b5", "b2", "b1", "b3"],
    ["b0", "b0", "b0", "b0", "b0", "b0", "b0", "b0"],
    ["", "", "", "", "", "", "", ""],
    ["", "", "", "", "", "", "", ""],
    ["", "", "", "", "", "", "", ""],
    ["", "", "", "", "", "", "", ""],
    ["w0", "w0", "w0", "w0", "w0", "w0", "w0", "w0"],
    ["w3", "w1", "w2", "w4", "w5", "w2", "w1", "w3"]
]   
    

# Load piece images
piece_image_paths = {
    "w0": "assets/w_pawn.svg",    # Pawn
    "w1": "assets/w_knight.svg",  # Knight
    "w2": "assets/w_bishop.svg",  # Bishop
    "w3": "assets/w_rook.svg",    # Rook
    "w4": "assets/w_queen.svg",   # Queen
    "w5": "assets/w_king.svg",    # King
    "b0": "assets/b_pawn.svg",    # Pawn
    "b1": "assets/b_knight.svg",  # Knight
    "b2": "assets/b_bishop.svg",  # Bishop
    "b3": "assets/b_rook.svg",    # Rook
    "b4": "assets/b_queen.svg",   # Queen
    "b5": "assets/b_king.svg"     # King
}

# Preload piece images
piece_images = {}
for piece, path in piece_image_paths.items():
    piece_images[piece] = Image.open(path).convert("RGBA")

# Function to create a chessboard image
def create_board_image(board):
    width = COLS * SQUARE_SIZE
    height = ROWS * SQUARE_SIZE
    board_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(board_img)
    
    # Iterate rows in reverse so that row 7 (white's home) is at the bottom
    for disp_row, board_row in enumerate(reversed(board)):
        for col in range(COLS):
            # disp_row now corresponds to the drawing row (0: bottom, 7: top)
            is_light_square = ((7 - disp_row) + col) % 2 == 0
            color = LIGHT_COLOR if is_light_square else DARK_COLOR
            
            x1 = col * SQUARE_SIZE
            # Update y coordinate based on disp_row (0 at bottom)
            y1 = disp_row * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
            piece_code = board_row[col]
            # Skip empty squares
            if piece_code.strip():
                if piece_code in piece_images:
                    piece_img = piece_images[piece_code].rotate(180, expand=True)
                    board_img.paste(piece_img, (x1, y1), piece_img)
                else:
                    print(f"Warning: {piece_code} image not found.")
    return board_img

# Function to handle chess game moves
def handle_move(game, from_square, to_square):
    if game.board.is_move_legal(from_square, to_square):
        game.board.move_piece(from_square, to_square)
        return True
    return False

def init_state():
    if "game" not in st.session_state:
        st.session_state.game = chessengine_py.Game()
    if "turn" not in st.session_state:
        st.session_state.turn = 0

# Streamlit UI
def main():
    init_state()
    game = st.session_state.game

    st.title("Chess Game")

    # Create a placeholder for the board image in the main area
    board_placeholder = st.empty()

    # Render the board in the main area
    board_state = game.board.get_board()
    img = create_board_image(board_state)
    board_placeholder.image(img, caption="Current Board", use_column_width=False)

    # Sidebar inputs for selecting piece and move
    input_piece = st.sidebar.text_input("Enter piece to move (e.g., 'e2'):", "")
    if input_piece:
        try:
            from_square = chessengine_py.Square(
                ord(input_piece[0]) - ord('a') + 8 * (int(input_piece[1]) - 1)
            )
            moves = game.board.legal_moves(from_square)
            legal_moves = [chessengine_py.square_to_string(move[1]) for move in moves]
            st.sidebar.write("Legal moves:", legal_moves)
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
                # Update the board image after the move
                board_state = game.board.get_board()
                img = create_board_image(board_state)
                board_placeholder.image(img, caption="Current Board", use_column_width=False)
            else:
                st.sidebar.error("Illegal move!")

    # Now, print game info (turn, en passant, check status, etc.) in the sidebar
    st.sidebar.write(f"Turn: {st.session_state.turn}")
    en_passant = game.board.get_en_passant_square()
    if en_passant != chessengine_py.Square.NO_SQUARE:
        st.sidebar.write(f"En Passant square: {chessengine_py.square_to_string(en_passant)}")

    if game.board.get_check(chessengine_py.Color.WHITE):
        st.sidebar.error("White is in check!")
    elif game.board.get_check(chessengine_py.Color.BLACK):
        st.sidebar.error("Black is in check!")

    all_legal_moves = []
    for square_int in range(64):
        square = chessengine_py.Square(square_int)
        all_legal_moves.extend(game.board.legal_moves(square))

    if not all_legal_moves and game.board.get_check(game.board.get_side_to_move()):
        winner = "Black" if game.board.get_side_to_move() == chessengine_py.Color.WHITE else "White"
        st.sidebar.write(f"Checkmate! {winner} wins!")

if __name__ == "__main__":
    main()
