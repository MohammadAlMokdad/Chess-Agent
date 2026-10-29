"""A fake chess GUI, used to test the vision stack without a screen.

Renders positions the way a web board does -- themed squares, light pieces
with dark outlines, dark pieces with light outlines, last-move highlights --
then runs the real classifier and move-inference code over thousands of
positions and checks every inferred move against the move actually played.
"""
from __future__ import annotations

import random

import chess
import cv2
import numpy as np

from chessagent import vision
from chessagent.config import Calibration

_HI_CONTRAST = dict(white_fill=(246, 246, 246), white_edge=(26, 26, 26),
                    black_fill=(34, 34, 36), black_edge=(215, 215, 215))

# chess.com's "neo" set, measured off a real board: the black pieces are a
# mid GREY (~89 luma), not near-black. That single fact is what makes a
# highlighted dark square dangerous -- lifted to ~185 it ends up *brighter*
# than the piece standing on it, so a square whose background tone is assumed
# rather than measured reads as the opposite colour.
_NEO = dict(white_fill=(249, 249, 249), white_edge=(61, 61, 61),
            black_fill=(89, 89, 92), black_edge=(36, 36, 38))

# light square BGR, dark square BGR, highlight BGR, piece palette.
THEMES = {
    "lichess-brown":  ((166, 209, 240), (86, 139, 181), (90, 215, 205), _HI_CONTRAST),
    "chesscom-green": ((208, 236, 238), (86, 149, 118), (95, 232, 248), _HI_CONTRAST),
    "chesscom-neo":   ((208, 236, 238), (86, 149, 118), (95, 232, 248), _NEO),
    "blue":           ((222, 227, 222), (150, 145, 106), (120, 200, 210), _HI_CONTRAST),
    "dark":           ((110, 110, 112), (64, 64, 66), (70, 150, 160), _HI_CONTRAST),
    "low-contrast":   ((205, 210, 206), (170, 176, 172), (150, 195, 200), _HI_CONTRAST),
}

# Tuned so the chess.com themes reproduce measured real-board behaviour:
# a highlight lifts a dark square by ~+50 luma and a light square by ~+1.
HIGHLIGHT_ALPHA = 0.5


def _piece_shape(kind: int, cx: int, cy: int, s: float) -> np.ndarray:
    """A crude but distinct silhouette per piece type."""
    if kind == chess.PAWN:
        pts = [(0, -0.42), (0.30, 0.10), (0.36, 0.42), (-0.36, 0.42), (-0.30, 0.10)]
    elif kind == chess.KNIGHT:
        pts = [(-0.10, -0.46), (0.34, -0.10), (0.30, 0.44), (-0.34, 0.44), (-0.26, 0.02)]
    elif kind == chess.BISHOP:
        pts = [(0, -0.48), (0.28, -0.02), (0.36, 0.44), (-0.36, 0.44), (-0.28, -0.02)]
    elif kind == chess.ROOK:
        pts = [(-0.36, -0.40), (0.36, -0.40), (0.30, -0.20), (0.34, 0.42), (-0.34, 0.42), (-0.30, -0.20)]
    elif kind == chess.QUEEN:
        pts = [(-0.42, -0.42), (-0.20, -0.10), (0, -0.44), (0.20, -0.10), (0.42, -0.42),
               (0.36, 0.44), (-0.36, 0.44)]
    else:  # king
        pts = [(-0.10, -0.50), (0.10, -0.50), (0.10, -0.34), (0.34, -0.34), (0.34, -0.14),
               (0.10, -0.14), (0.36, 0.44), (-0.36, 0.44), (-0.10, -0.14), (-0.34, -0.14),
               (-0.34, -0.34), (-0.10, -0.34)]
    return np.array([[int(cx + x * s), int(cy + y * s)] for x, y in pts], dtype=np.int32)


def render(board: chess.Board, playing_white: bool = True, theme: str = "lichess-brown",
           cell: int = 72, highlight: tuple | None = None, noise: float = 0.0) -> np.ndarray:
    light, dark, hl, palette = THEMES[theme]
    size = cell * 8
    img = np.zeros((size, size, 3), dtype=np.uint8)

    hl_cells = set()
    if highlight is not None:
        for sq in highlight:
            hl_cells.add(vision.square_to_screen(sq, playing_white))

    for r in range(8):
        for c in range(8):
            base = light if (r + c) % 2 == 0 else dark
            # Last-move highlights are an alpha blend toward a bright colour,
            # not a flat replacement. That asymmetry matters: measured on a
            # real chess.com board, the same highlight lifts a dark square by
            # +52 luma but a light square by only +1, because the highlight
            # tone sits right next to the light-square tone. A square's stored
            # base tone is therefore badly wrong exactly when it is
            # highlighted -- which is the case worth testing.
            colour = tuple(int((1 - HIGHLIGHT_ALPHA) * b + HIGHLIGHT_ALPHA * h)
                           for b, h in zip(base, hl)) if (r, c) in hl_cells else base
            img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = colour

    for r in range(8):
        for c in range(8):
            piece = board.piece_at(vision.screen_to_square(r, c, playing_white))
            if piece is None:
                continue
            fill, edge = (palette["white_fill"], palette["white_edge"]) \
                if piece.color == chess.WHITE \
                else (palette["black_fill"], palette["black_edge"])
            cx, cy = int((c + 0.5) * cell), int((r + 0.5) * cell)
            poly = _piece_shape(piece.piece_type, cx, cy, cell * 0.78)
            cv2.fillPoly(img, [poly], fill, lineType=cv2.LINE_AA)
            # Edge and detail weights are matched to a measured real board
            # (lichess cburnett): a white piece reads ~39% bright / ~23% dark,
            # a black piece ~2% bright / ~58% dark. Getting this ratio right
            # matters -- it is exactly the signal the colour classifier uses,
            # so a renderer that gets it wrong tests the wrong thing.
            if piece.color == chess.WHITE:
                cv2.polylines(img, [poly], True, edge, max(2, cell // 28), lineType=cv2.LINE_AA)
                lw = max(1, cell // 38)
                for frac, wide in ((0.36, 0.26), (0.20, 0.18)):
                    y = int(cy + frac * cell * 0.78)
                    half = int(cell * 0.78 * wide)
                    cv2.line(img, (cx - half, y), (cx + half, y), edge, lw, cv2.LINE_AA)
            else:
                cv2.polylines(img, [poly], True, edge, max(1, cell // 48), lineType=cv2.LINE_AA)

    if noise > 0:
        img = np.clip(img.astype(np.float32) +
                      np.random.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def _calibrate_for(theme: str, playing_white: bool, cell: int, noise: float) -> Calibration:
    img = render(chess.Board(), playing_white, theme, cell, noise=noise)
    return Calibration(left=0, top=0, size=img.shape[0], **vision.learn_from_start(img))


def run_selftest(games: int = 20, verbose: bool = True) -> int:
    rng = random.Random(7)
    totals = {"moves": 0, "ok": 0, "squares_bad": 0, "fail": []}

    combos = [(t, pw, cell, noise)
              for t in THEMES
              for pw in (True, False)
              for cell, noise in ((72, 0.0), (48, 2.5), (96, 4.0))]

    for theme, pw, cell, noise in combos:
        calib = _calibrate_for(theme, pw, cell, noise)

        # Orientation must be recovered from pixels alone.
        if calib.playing_white != pw:
            totals["fail"].append(f"{theme}/{cell}px: orientation detected wrong")
            continue

        for _ in range(max(1, games // len(combos) + 1)):
            board = chess.Board()
            last = None
            for _ply in range(rng.randint(10, 60)):
                if board.is_game_over():
                    break
                move = rng.choice(list(board.legal_moves))

                board.push(move)
                img = render(board, pw, theme, cell,
                             highlight=(move.from_square, move.to_square), noise=noise)
                board.pop()

                grid, _ = vision.read_grid(img, calib)
                truth_board = board.copy()
                truth_board.push(move)
                truth = vision.render_grid(truth_board, pw)
                bad = int((grid != truth).sum())
                totals["squares_bad"] += bad

                got, info = vision.infer_move(board, grid, pw)
                totals["moves"] += 1
                # Promotions collapse to a queen by design; judge accordingly.
                want = move
                if move.promotion is not None:
                    want = chess.Move(move.from_square, move.to_square, chess.QUEEN)
                if got == want:
                    totals["ok"] += 1
                else:
                    totals["fail"].append(
                        f"{theme}/{cell}px/{'W' if pw else 'B'}: "
                        f"played {move.uci()}, read {got.uci() if got else info} "
                        f"({bad} bad squares)")
                board.push(move)
                last = move

    n, ok = totals["moves"], totals["ok"]
    if verbose:
        print(f"Themes x orientations x scales tested: {len(combos)}")
        print(f"Positions checked: {n}")
        print(f"Moves correctly inferred: {ok}/{n}  ({100.0 * ok / max(n, 1):.2f}%)")
        print(f"Square misreads: {totals['squares_bad']} of {n * 64} "
              f"({100.0 * totals['squares_bad'] / max(n * 64, 1):.4f}%)")
        for line in totals["fail"][:15]:
            print("  FAIL", line)
        if len(totals["fail"]) > 15:
            print(f"  ... and {len(totals['fail']) - 15} more")

    return 0 if ok == n and not totals["fail"] else 1
