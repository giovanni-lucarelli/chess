#!/usr/bin/env python3
from __future__ import annotations

# stdlib
import os
from io import BytesIO
from typing import Iterable, Optional

import matplotlib.pyplot as plt  
from PIL import Image  # per mostrare inline quando non salviamo su file

# chess
import chess
import chess.svg
from cairosvg import svg2png

def _render_svg_from_fen(
    fen: str,
    *,
    flipped: bool = False,
    size: int = 640,
    coordinates: bool = True,
    lastmove: Optional[str] = None,          # es. "e2e4"
    arrows: Optional[Iterable[str]] = None,  # es. ["e2e4","g1f3"]
    squares: Optional[Iterable[str]] = None, # es. ["e4","d5"]
) -> str:
    board = chess.Board(fen)

    lm = None
    if lastmove:
        try:
            lm = chess.Move.from_uci(lastmove)
        except Exception:
            lm = None 

    # arrows
    arrow_objs = []
    if arrows:
        for a in arrows:
            try:
                u = a.strip().lower()
                u = u.replace("-", "").replace(" ", "")
                if len(u) >= 4:
                    a_from = chess.parse_square(u[:2])
                    a_to = chess.parse_square(u[2:4])
                    arrow_objs.append(chess.svg.Arrow(a_from, a_to))
            except Exception:
                pass

    # highlighted squares
    sq_list = []
    if squares:
        for s in squares:
            try:
                sq_list.append(chess.parse_square(s))
            except Exception:
                pass

    # generate SVG
    svg = chess.svg.board(
        board=board,
        size=size,
        coordinates=coordinates,
        flipped=flipped,
        lastmove=lm,
        arrows=arrow_objs,
        squares=sq_list,
    )

    return svg

def plot_fen(
    fen: str,
    ax=None,
    title: Optional[str] = None,
    flipped: bool = False,
    *,
    size: int = 640,
    coordinates: bool = True,
    lastmove: Optional[str] = None,
    arrows: Optional[Iterable[str]] = None,
    squares: Optional[Iterable[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Disegna la scacchiera della FEN.
    - Se save_path è fornito:
        - .svg -> salva SVG
        - altrimenti -> salva PNG (via cairosvg)
    - Se save_path è None, mostra il PNG inline con matplotlib.
    """
    svg_str = _render_svg_from_fen(
        fen,
        flipped=flipped,
        size=size,
        coordinates=coordinates,
        lastmove=lastmove,
        arrows=arrows,
        squares=squares,
    )

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if save_path.lower().endswith(".svg"):
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(svg_str)
        else:
            svg2png(bytestring=svg_str.encode("utf-8"), write_to=save_path, dpi=300)
        return None

    png_bytes = svg2png(bytestring=svg_str.encode("utf-8"))
    img = Image.open(BytesIO(png_bytes))
    if ax is None:
        fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax

def plot_game(
    game,
    save_path: Optional[str] = None,
    **kwargs,
):
    """
    Disegna/salva la posizione corrente di un Game (usa game.to_fen()).
    Accetta gli stessi kwargs di plot_fen (size, flipped, coordinates, lastmove, arrows, squares).
    """
    fen = game.to_fen()
    return plot_fen(fen, save_path=save_path, **kwargs)

if __name__ == "__main__":
    fen = input("Insert FEN: ")
    plot_fen(fen, title="Preview", flipped=False, size=640)
    plt.show()
