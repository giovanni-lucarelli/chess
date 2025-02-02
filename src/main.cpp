// #include "bitboard.hpp"
// #include "chessboard.hpp"
// #include "types.hpp"
// #include <iostream>

// int main() {

//     /* ---------------------------------------------------------------------- */
//     /*                   Bitboard Representation Of A Board                   */
//     /* ---------------------------------------------------------------------- */

//     // // Initialize an empty bitboard
//     // U64 board = 0;

//     // // Set a piece on B1
//     // Bitboard::set_bit(board, B1);
//     // Bitboard::print(board);
//     // Bitboard::get_bit(board, B1) ? std::cout << "B1 is set\n" : std::cout << "B1 is not set\n";
//     // Bitboard::clear_bit(board, B1);
//     // Bitboard::get_bit(board, B1) ? std::cout << "B1 is set\n" : std::cout << "B1 is not set\n";

//     // // Set pieces in rank 2
//     // for (int sq = A2; sq <= H2; sq++) {
//     //     Bitboard::set_bit(board, static_cast<Square>(sq));
//     // }

//     // // Check total set pieces and print board
//     // std::cout << "Total set pieces: " << Bitboard::count_bits(board) << "\n";
//     // Bitboard::print(board);

//     /* ---------------------------------------------------------------------- */

//     // Initialize a chessboard
//     ChessBoard board2;

//     // test get_pieces
//     U64 wp{board2.get_pieces(WHITE, PAWN)};
//     Bitboard::print(wp);

//     // Print chessboard
//     board2.print();

//     // Check if a square is occupied
//     board2.is_occupied(D8) ? std::cout << "D8 is occupied\n" : std::cout << "D8 is not occupied\n";

//     // Get piece on a square
//     auto [color, piece] = board2.get_piece_on_square(D8);
//     std::cout << "Piece on D8: " << color_to_string(color) << " " << piece_to_string(piece) << "\n";
    
//     try
//     {
//         auto [color2, piece2] = board2.get_piece_on_square(D3);
//         std::cout << "Piece on D3: " << color_to_string(color2) << " " << piece_to_string(piece2) << "\n";
//     }
//     catch(const std::exception& e)
//     {
//         std::cerr << e.what() << '\n';
//     }
    
    
//     // Move a piece
//     board2.move_piece(D2, D4);
//     board2.print();

//     // print whos turn is
//     std::cout << "Side to move: " << color_to_string(board2.get_side_to_move()) << "\n";

//     board2.move_piece(D7, D6);
//     board2.print();

//     board2.move_piece(C1, F3);
//     board2.print();
//     return 0;
// }


#include "game.hpp"

int main() {
    Game game;
    game.start1v1();
    return 0;
}
