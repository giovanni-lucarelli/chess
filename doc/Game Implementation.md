# Game Implementation

The following document contains details about the implementation of the chess game in C++.

The code has been organized in increasing order of abstraction:
- `types.hpp`: basic custom C++ types, usefull for the implementation
- `bitboard.hpp`: utilities for the bitboard representation of the chessboard [(see wikipedia)](https://en.wikipedia.org/wiki/Bitboard), *i.e.*, efficient implementation of the chessboard using `int64`
- `chessboard.hpp`: actual chessboard that use the bitboard implementation. It enables to set/get a position, remove pieces etc
- `move.hpp`: declaration of the `Move` class that stores all the important information of a move that will be used to update the state of the game and to check the relative rules(type, from/to, captured pieces, promoted piece, and so on)
- `game.hpp`: declaration of `Game` class that stores the state of the game, check the rules of each move, execute the move (dynamic of the game).

## Custom types and bitboard representation

A clever and efficient idea in order to store a chessboard state is to use the bitboard representation. In this representation, each color and each kind of piece (pawn, knight, bishop, rook, queen, king) correspond to a 64-bit integer. This is possible since the chessboard is 8x8, therefore we can denote the presence (or the absence) of a piece with a 1 (or a 0) in the correct position. 

For example the two white knight in the starting position (B1, G1) can be represented by:

    0 0 0 0 0 0 0 0  : Rank 8
    0 0 0 0 0 0 0 0  : Rank 7
    0 0 0 0 0 0 0 0  : Rank 4
    0 0 0 0 0 0 0 0  : Rank 6
    0 0 0 0 0 0 0 0  : Rank 5
    0 0 0 0 0 0 0 0  : Rank 3
    0 0 0 0 0 0 0 0  : Rank 2
    0 1 0 0 0 0 1 0  : Rank 1  

eq. in binary: 

    00000000 00000000 00000000 00000000 00000000 00000000 01000010

eq. in decimal: `66`, or in hexadecimal `0x42`. 

> **Note:** Hexadecimal representation is usefull since the binary conversion is very simple. Each hex digit is exactly 4 bits: 0 = 0000, F = 1111.

For clarity in the indexing (for colors, pieces, squares), `enum` types have been defined (see `types.hpp`):    

```{C}
enum Color { WHITE, BLACK, NO_COLOR };

enum Piece { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE };

enum Square {
  A1, B1, C1, D1, E1, F1, G1, H1,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A8, B8, C8, D8, E8, F8, G8, H8,
  NO_SQUARE
};
```

## Chessboard class

The chessboard class aim to represent the physical chessboard:
- stores pieces currently placed on it
- has the ability to add or remove pieces

Specifically, as mentioned above, the pieces on the board are stored as:

    using U64 = uint64_t
    std::array<std::array<U64, 6>, 2> pieces;   

where the row index denotes the color, *e.g.* WHITE, (passed through the `Color` enum) and the column index denotes the chess piece, *e.g.* KNIGHT, (`Piece` enum).


> **Example:** from `chessboard.cpp` 
>
> `pieces[WHITE][KNIGHT] = 0x0000000000000042ULL; // B1, G1`
>
> for visual clarity (16 hex digits is eq. to 64 bits, so every pair of hex digits starting from right represents a chessboard rank) and ULL to specify the unsigned long long (64-bit integer)

## Game class

The game class represents a chess game, specifically:
- stores the game state, defined by `board` (containing the pieces on the board), `side_to_move`, `en_passant_square`, `castling_rights`, `white_check`, `black_check`, `checkmate` (terminal state)
- check the validity of a move (passed as a `Move` object, see `move.hpp`)
- do and undo the move
- manage a game between two players or against the computer
- initialize a new game based on the Forsyth–Edwards Notation (FEN)

> **Remark:** the Forsyth–Edwards Notation (FEN) is a standard notation for describing a particular board position of a chess game.

## Move class

The move class is a simpe class used to specify the attributes of a move. A chess move is parsed to the program using the traditional notation: 
    
    from_square to_square

The constructor of Move class automatically defines the attributes of a move (e.g. capture, promotion, check, ...), these are usefull in order to write self-documented code, specifically in the rules-checking functions.