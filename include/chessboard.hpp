#pragma once
#include "types.hpp"
#include "bitboard.hpp"
#include <array>
#include <string>
#include <vector>
#include <set>

class ChessBoard {
private:

    // Bitboards for [Color][Piece]
    std::array<std::array<U64, 6>, 2> pieces;

public:

    // constructor 
    ChessBoard(){clear();};

/* --------------------------------- Getter --------------------------------- */

    std::vector<std::vector<std::string>> get_board() const;

    // Get all pieces of a type/color
    U64 get_pieces(Color color, Piece piece) const;

    // return the piece and the color on a square
    std::pair<Color, Piece> get_piece_on_square(Square sq) const;

        
/* --------------------------------- Setter --------------------------------- */
    
    bool add_piece(Color color, Piece piece, Square sq);
    void remove_piece(Square sq);

    void clear() {
        for (auto& color_pieces : pieces) {
        color_pieces.fill(0);
        }
    };

    // set the initial traditional chess position
    void reset();


/* -------------------------------- Utilities ------------------------------- */

    bool is_path_clear(Square from, Square to) const;
    bool is_occupied(Square sq) const;
    // A helper: convert a square index to row and column.
    void square_to_coord(Square sq, int &row, int &col) const {
        row = sq / 8;
        col = sq % 8;
    }
    void print() const;

   
};