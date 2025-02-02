// #include "movegen.hpp"
// #include "types.hpp"
// #include <iostream>

// std::vector<Move> MoveGenerator::generate_moves(const ChessBoard& board) {
//     std::vector<Move> moves;
//     generate_pawn_moves(board, moves);
//     // Add other piece move generation logic here
//     return moves;
// }

// void MoveGenerator::generate_pawn_moves(const ChessBoard& board, std::vector<Move>& moves) {
//     Color side_to_move = board.get_side_to_move();
//     int direction = (side_to_move == WHITE) ? 8 : -8;
//     int start_rank = (side_to_move == WHITE) ? 1 : 6;
//     int promotion_rank = (side_to_move == WHITE) ? 6 : 1;
//     int en_passant_rank = (side_to_move == WHITE) ? 4 : 3;

//     U64 pawns = board.get_pieces(side_to_move, PAWN);

//     while (pawns) {
//         Square from = static_cast<Square>(__builtin_ctzll(pawns));
//         U64 from_bit = 1ULL << from;

//         // Remove the pawn from the bitboard
//         pawns &= ~from_bit;

//         // Add pawn single moves
//         Square to = static_cast<Square>(from + direction); // Move forward
//         if (!board.is_occupied(to)) {
//             moves.push_back(Move(from, to, PAWN, board.get_piece_on_square(to).second));
//             std::cout << "Single move: " << square_to_string(from) << " to " << square_to_string(to) << std::endl;
//         }

//         // Add pawn captures
//         // Capture left
//         if (from % 8 != 0) { // Ensure not on the 'a' file
//             to = static_cast<Square>(from + direction - 1);
//             // std::cout << "Checking capture left from " << square_to_string(from) << " to " << square_to_string(to) << std::endl;
//             if (board.is_occupied(to) && board.get_piece_on_square(to).first != side_to_move) {
//                 moves.push_back(Move(from, to));
//                 std::cout << "Capture left: " << square_to_string(from) << " to " << square_to_string(to) << std::endl;
//             }
//         }
//         // Capture right
//         if (from % 8 != 7) { // Ensure not on the 'h' file
//             to = static_cast<Square>(from + direction + 1);
//             // std::cout << "Checking capture right from " << square_to_string(from) << " to " << square_to_string(to) << std::endl;
//             if (board.is_occupied(to) && board.get_piece_on_square(to).first != side_to_move) {
//                 moves.push_back(Move(from, to));
//                 std::cout << "Capture right: " << square_to_string(from) << " to " << square_to_string(to) << std::endl;
//             }
//         }

//         // Add pawn double moves
//         if ((side_to_move == WHITE && from / 8 == 1) || (side_to_move == BLACK && from / 8 == 6)) {
//             to = static_cast<Square>(from + 2 * direction); // Move forward two squares
//             if (!board.is_occupied(to) && !board.is_occupied(static_cast<Square>(from + direction))) {
//                 moves.push_back(Move(from, to));
//                 std::cout << "Double move: " << square_to_string(from) << " to " << square_to_string(to) << std::endl;
//             }
//         }
//     }
// }


