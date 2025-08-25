# Dataset Info

The tablebase solution are divided by endgame type:
- `kk.csv`: king vs. king (trivial) endgame 
- `krk/`: files relative to king and rook vs. king endgame
- `kqk/`: files relative to king and queen vs. king endgame
- `kbbk/`: files relative to king and bishop pair vs. king endgame

All the folders above contain:
- `*_full.csv`: dataset storing `fen | side_to_move | wdl | dtz`, both side to move
- `*_train.csv`: dataset for training (0.8 subset of full), white to move
- `*_test.csv`: dataset for testing (0.1 subset of full), white to move
- `*_val.csv`: dataset for validation (0.1 subset of full), white to move
- `*.rtbw`: tablebase file used by python-chess.syzygy storing win/draw/loss information
- `*.rtbz`: tablebase file used by python-chess.syzygy storing distance-to-zero information for access at the root.