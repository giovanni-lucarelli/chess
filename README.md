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


