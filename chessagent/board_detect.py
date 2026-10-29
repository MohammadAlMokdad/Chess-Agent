"""Locate a chessboard on screen by hunting for an 8x8 checker pattern.

The signal we look for is simple and theme-independent: split a candidate
square region into an 8x8 grid, and ask how cleanly the 64 cell-means split
into two tight clusters by colour parity. A real board scores enormously
higher than anything else on a desktop.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import capture


def _cell_means(integral: np.ndarray, ys: np.ndarray, xs: np.ndarray, size: int) -> np.ndarray:
    """-> (64, len(ys), len(xs)) array of mean intensity per grid cell."""
    c = size / 8.0
    out = np.empty((64, len(ys), len(xs)), dtype=np.float32)
    for i in range(8):
        y1 = ys + int(round(i * c))
        y2 = ys + int(round((i + 1) * c))
        for j in range(8):
            x1 = xs + int(round(j * c))
            x2 = xs + int(round((j + 1) * c))
            area = float((y2[0] - y1[0]) * (x2[0] - x1[0])) or 1.0
            block = (
                integral[np.ix_(y2, x2)]
                - integral[np.ix_(y1, x2)]
                - integral[np.ix_(y2, x1)]
                + integral[np.ix_(y1, x1)]
            )
            out[i * 8 + j] = block / area
    return out


_PARITY = np.array([[(i + j) % 2 for j in range(8)] for i in range(8)]).ravel().astype(bool)


def _score(cells: np.ndarray) -> np.ndarray:
    """Checkerboard plausibility for every candidate position."""
    a = cells[~_PARITY]          # one colour of square
    b = cells[_PARITY]           # the other
    separation = np.abs(a.mean(0) - b.mean(0))
    noise = a.std(0) + b.std(0)
    return separation / (noise + 4.0)


def _search(gray: np.ndarray, sizes, stride: int):
    """Best (score, left, top, size) over a grid of candidates."""
    integral = cv2.integral(gray).astype(np.float64)
    h, w = gray.shape
    best = (-1.0, 0, 0, 0)
    for size in sizes:
        if size + 1 > min(h, w):
            continue
        ys = np.arange(0, h - size, stride, dtype=np.intp)
        xs = np.arange(0, w - size, stride, dtype=np.intp)
        if len(ys) == 0 or len(xs) == 0:
            continue
        scores = _score(_cell_means(integral, ys, xs, size))
        idx = int(np.argmax(scores))
        iy, ix = divmod(idx, len(xs))
        if scores[iy, ix] > best[0]:
            best = (float(scores[iy, ix]), int(xs[ix]), int(ys[iy]), int(size))
    return best


def find_board(min_size: int = 200, max_size: int | None = None):
    """Scan all monitors. Returns (left, top, size, confidence) in screen pixels."""
    screen = capture.virtual_screen()
    hit = find_board_in(capture.grab(screen), min_size, max_size)
    if hit is None:
        return None
    x, y, size, score = hit
    return (screen["left"] + x, screen["top"] + y, size, score)


def find_board_in(frame: np.ndarray, min_size: int = 200, max_size: int | None = None):
    """Same search against a supplied image. Returns (x, y, size, confidence)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    # Coarse pass on a downscaled copy: cheap, gets us within a few pixels.
    k = max(1, int(round(gray.shape[1] / 640.0)))
    small = cv2.resize(gray, (gray.shape[1] // k, gray.shape[0] // k), interpolation=cv2.INTER_AREA)
    lo = max(48, min_size // k)
    hi = min(min(small.shape) - 2, (max_size // k) if max_size else min(small.shape) - 2)
    if hi <= lo:
        return None
    coarse = _search(small, range(lo, hi, 4), stride=3)
    if coarse[0] < 0:
        return None

    # Fine pass at full resolution, in a small window around the coarse hit.
    _, cx, cy, cs = coarse
    cx, cy, cs = cx * k, cy * k, cs * k
    pad = 3 * k
    y0, x0 = max(0, cy - pad), max(0, cx - pad)
    patch = gray[y0:cy + cs + pad, x0:cx + cs + pad]
    sizes = range(max(64, cs - 2 * k), cs + 2 * k + 1)
    fine = _search(patch, sizes, stride=1)
    if fine[0] < 0:
        return None

    score, fx, fy, fs = fine
    return (x0 + fx, y0 + fy, fs, score)
