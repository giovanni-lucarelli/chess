## How to run

```bash

cmake -B build
cmake --build build
cd build
./ChessEngine

```

## How to test

```bash

cmake -B build
cmake --build build
cd build
./ChessEngineTest

```

## Roadmap

1. Setup the game in order to play one vs one
   1. C++
   2. Python bindings and interface
2. Create the MoveGenerator class for all the pieces. It should be able to generate all the possible moves for a given position. This will be used to implement the AI.
3. Implement the AI using the minimax algorithm with alpha-beta pruning.
4. Implement the UI in python using the bindings created in step 1.
5. Play online

## Project structure

```
myChess_Engine/
├── .gitignore
├── chess.md
├── CMakeLists.txt
├── include/
│   ├── bitboard.hpp
│   ├── chessboard.hpp
│   ├── game.hpp
│   ├── move.hpp
│   ├── movegen.hpp
│   └── types.hpp
├── README.md
├── src/
│   ├── chessboard.cpp
│   ├── game.cpp
│   ├── main.cpp
│   └── movegen.cpp
├── test/
│   └── test.cpp
└── todo.md
```

The chess program is structured in the following way:

- `types.hpp` contains the definition of the `Piece` and `Color` enums, which are used to represent the pieces and colors in the game.
- `bitboard.hpp` contains the definition of the `Bitboard` class, which is used to represent the state of the board.
- `chessboard.hpp` contains the definition of the `Chessboard` class, which is used to represent the state of the game and the rules of chess.
- `game.hpp` contains the definition of the `Game` class, which is used to manage the game.
- `move.hpp` contains the definition of the `Move` class, which is used to represent a move.
- `movegen.hpp` contains the definition of the `MoveGenerator` class, which is used to generate all the possible moves for a given position. This will be used to implement the AI.

In particular, each position on the board is represented by a 64-bit integer, which is used as a bitboard to store the state of the board. The `Bitboard` class provides methods to set and get the state of a given square, as well as to perform bitwise operations on the bitboards (e.g., adding, removing, or checking pieces). 

The `Chessboard` class uses an 3 dimensional array to represent the state of the board, with the first dimension representing the color of the piece (white or black), the second dimension representing the type of the piece (pawn, knight, bishop, rook, queen, king), and the third dimension representing the position of the piece on the board. In this class are also stored the chess rules through `is_legal_move` method.

The `Game` class manages the game, including the current state of the board, the current player, and the history of the moves. The `Move` class represents a move, including the starting and ending squares, the piece that is moved, and any captured piece. The `MoveGenerator` class generates all the possible moves for a given position, including pawn moves, knight moves, bishop moves, rook moves, queen moves, and king moves.



