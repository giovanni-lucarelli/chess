import sys 
sys.path.append("build")
import chessengine_py # type: ignore

def play():
    turn = 0
    game = chessengine_py.Game()
    
    while True:
        turn += 1
        game.board.print()
        
        # Check if the king is in check
        game.board.check_control()
        if game.board.get_check(chessengine_py.Color.WHITE):
            print("\033[1;31mWhite in check!\033[0m")
        elif game.board.get_check(chessengine_py.Color.BLACK):
            print("\033[1;31mBlack in check!\033[0m")
        
        # Printing en passant square
        if game.board.get_en_passant_square() != chessengine_py.Square.NO_SQUARE:
            print("\nEn passant square:", chessengine_py.square_to_string(game.board.get_en_passant_square()))
        
        # Checkmate detection
        side_to_move = game.board.get_side_to_move()
        all_legal_moves = []
        for square_int in range(64):  # Iterate over all board squares
            square = chessengine_py.Square(square_int)  # Convert int to Square enum
            all_legal_moves.extend(game.board.legal_moves(square))
            if square != chessengine_py.Square.NO_SQUARE:  # Ignore invalid square
                all_legal_moves.extend(game.board.legal_moves(square))



        if not all_legal_moves and game.board.get_check(side_to_move):
            winner = "Black" if side_to_move == chessengine_py.Color.WHITE else "White"
            print(f"\033[1;31mCheckmate! {winner} wins!\033[0m")
            break
        
        # Printing turn and color to move
        if side_to_move == chessengine_py.Color.WHITE:
            print(f"\033[1;34mWhite to move ({(turn // 2) + 1})\033[0m")
        else:
            print(f"\033[1;34mBlack to move ({turn // 2})\033[0m")
        
        # Getting user input
        input_piece = input("Enter piece to move (only its square): ")
        if input_piece.lower() == "exit":
            break
        
        from_square = chessengine_py.Square(ord(input_piece[0]) - ord('a') + 8 * (int(input_piece[1]) - 1))
        
        # Printing legal moves
        moves = game.board.legal_moves(from_square)
        print("Legal moves:", [chessengine_py.square_to_string(move[1]) for move in moves])
        
        input_move = input("Enter move (only its square): ")
        if input_move.lower() == "exit":
            break
        
        move = game.parse_input(input_piece, input_move)
        
        # Check if move is legal
        if not game.board.is_move_legal(from_square, move[1]):
            print("Illegal move")
            continue
        
        # Make the move
        game.board.move_piece(from_square, move[1])

if __name__ == "__main__":
    play()