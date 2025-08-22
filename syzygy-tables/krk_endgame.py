# krk_endgame.py
import csv, chess, chess.syzygy

TB_DIR = "./"

def kings_ok(wk, bk):
    ax, ay = wk % 8, wk // 8
    bx, by = bk % 8, bk // 8
    return max(abs(ax - bx), abs(ay - by)) > 1  # not adjacent

def board_from(wk, bk, wr, white_to_move):
    b = chess.Board(None)  # empty board
    b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    b.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
    b.turn = chess.WHITE if white_to_move else chess.BLACK
    b.clear_stack()  # sets halfmove=0, fullmove=1
    return b

with chess.syzygy.open_tablebase(TB_DIR) as tb, open("krk_dtz.csv","w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fen","wdl","dtz"])

    for wk in range(64):
        for bk in range(64):
            if bk == wk or not kings_ok(wk, bk):
                continue
            for wr in range(64):
                if wr == wk or wr == bk:
                    continue
                for white_to_move in (True, False):
                    b = board_from(wk, bk, wr, white_to_move)
                    if not b.is_valid():
                        continue  # safety
                    try:
                        wdl = tb.probe_wdl(b)  # returns -2..+2
                        dtz = tb.probe_dtz(b)  # plies (DTZ50)
                        w.writerow([b.fen(), wdl, dtz])
                    except KeyError:
                        # illegal/unhandled (should be rare with the filters above)
                        pass

print("Wrote krk_dtz.csv")
