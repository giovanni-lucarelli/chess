# utils/plot_chess.py
import matplotlib.pyplot as plt

UNICODE = {
    'K':'♔','Q':'♕','R':'♖','B':'♗','N':'♘','P':'♙',
    'k':'♚','q':'♛','r':'♜','b':'♝','n':'♞','p':'♟'
}

def fen_to_grid(fen: str):
    board_fen = fen.split()[0]
    rows = board_fen.split('/')
    grid = []
    for r in rows:
        row = []
        for ch in r:
            if ch.isdigit():
                row.extend([' '] * int(ch))
            else:
                row.append(ch)
        grid.append(row)   # row 0 = rank 8
    return grid

def plot_fen(fen: str, ax=None, title=None, flipped=False):
    grid = fen_to_grid(fen)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    ax.set_aspect('equal')

    # draw board
    for r in range(8):
        for c in range(8):
            rr = r if not flipped else 7 - r
            cc = c if not flipped else 7 - c
            ax.add_patch(plt.Rectangle((c, 7 - r), 1, 1, fill=True,
                                       facecolor=('#EEE' if (r+c)%2==0 else '#888')))
            piece = grid[rr][cc]
            if piece.strip():
                ax.text(c+0.5, 7-r+0.5, UNICODE.get(piece, '?'),
                        ha='center', va='center', fontsize=28)

    # coords
    files = "abcdefgh" if not flipped else "hgfedcba"
    ranks = "12345678" if not flipped else "87654321"
    for i,f in enumerate(files):
        ax.text(i+0.5, -0.15, f, ha='center', va='top', fontsize=10)
    for i,rk in enumerate(ranks):
        ax.text(-0.15, i+0.5, rk, ha='right', va='center', fontsize=10)

    ax.set_xlim(0,8); ax.set_ylim(0,8)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title)
    plt.tight_layout()
    return ax

def plot_game(game, **kwargs):
    """Convenience: plot current position of a Game."""
    return plot_fen(game.to_fen(), **kwargs)
