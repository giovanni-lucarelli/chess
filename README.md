First idea: use `enum` data structure to map all the squares of the board (A1,...,H8) into integers. Then, store all the empty/non empty using a 64 bit integer i.e. a bitboard.
The choice of using `enum` is aimed to easly set some position.

## How to build

```bash

cmake -B build
cmake --build build

```

Run:

```bash

cd build
./ChessEngine

```

## Roadmap

1. Setup the game in order to play one vs one
   1. C++
   2. Python bindings and interface
2. Create the MoveGenerator class for all the pieces. It should be able to generate all the possible moves for a given position. This will be used to implement the AI.
3. Implement the AI using the minimax algorithm with alpha-beta pruning.
4. Implement the UI in python using the bindings created in step 1.
5. Play online



