"""Reading a chess position off the screen.

Design notes
------------
We deliberately do NOT try to recognise piece *types* from pixels. We only
classify each square as empty / white-occupied / black-occupied, then ask
which legal move would produce that pattern. The legal-move list does the
heavy lifting, so the vision layer solves a 3-class problem instead of a
13-class one, and it can never invent an illegal move.

The 3-class classifier trains itself on the starting position, where the
labels are known for free: outer two ranks occupied, middle four empty.
"""
from __future__ import annotations

import chess
import cv2
import numpy as np

from .config import (EMPTY, WHITE, BLACK, INNER_CROP, RING_INNER, RING_OUTER,
                     MIN_MATCH, MIN_MARGIN, LABELS)

FEAT_DIM = 5
FEAT_VERSION = 5   # bump whenever the feature vector changes; invalidates saved calibration

# Per-dimension weights on the distance metric, folded into `scale` at
# calibration time. Nearest-centroid weights every axis equally by default,
# which is wrong here: the colour-lean dimension is one opinion about piece
# colour, while the four before it are four largely independent opinions
# about occupancy. At full weight colour starts overruling them and accuracy
# falls off; held at 0.75 it sharpens colour calls without competing.
# Measured on the full test matrix plus a real lichess board: no loss
# anywhere, and the weakest real square's margin improves from 0.14 to ~0.33.
FEAT_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 0.75], dtype=np.float32)


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------
def _luma(board_img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _squares(lum: np.ndarray):
    """Yield (row, col, inner_crop, background) for all 64 squares.

    `background` is the square's own empty tone *in this frame*, taken as the
    median of a ring around where a piece sits. Measuring it per frame rather
    than trusting a value stored at calibration is what makes reading robust
    to last-move highlights, hover effects and check indicators -- they tint
    the ring exactly as much as they tint the middle.

    The median matters: coordinate labels are drawn in square corners, and a
    median ignores them where a mean would not.
    """
    cell = lum.shape[0] / 8.0

    def edges(row, col, frac):
        m = cell * (1.0 - frac) / 2.0
        return (int(round(row * cell + m)), int(round((row + 1) * cell - m)),
                int(round(col * cell + m)), int(round((col + 1) * cell - m)))

    for r in range(8):
        for c in range(8):
            iy0, iy1, ix0, ix1 = edges(r, c, INNER_CROP)
            oy0, oy1, ox0, ox1 = edges(r, c, RING_OUTER)
            ry0, ry1, rx0, rx1 = edges(r, c, RING_INNER)
            ring = np.concatenate([
                lum[oy0:ry0, ox0:ox1].ravel(),   # above the piece
                lum[ry1:oy1, ox0:ox1].ravel(),   # below
                lum[ry0:ry1, ox0:rx0].ravel(),   # left
                lum[ry0:ry1, rx1:ox1].ravel(),   # right
            ])
            inner = lum[iy0:iy1, ix0:ix1]
            bg = float(np.median(ring)) if ring.size else float(np.median(inner))
            yield r, c, inner, bg


def _headroom(value: float, base: float) -> float:
    """Deviation from `base`, as a fraction of the room available that way.

    A raw difference is the obvious choice and it is wrong under a highlight.
    A highlight lifts a square's *background* without touching the piece
    standing on it, so a white piece that cleared its background by 114 luma
    now clears it by only 71 -- and its dark outlines, unchanged, plunge
    further below. Both numbers slide toward the black-piece signature even
    though nothing about the piece changed.

    Measuring against the headroom instead is stable: that same white piece
    uses 93% of the room above its background unhighlighted, and 90% of it
    highlighted. Returns -1..+1.
    """
    if value >= base:
        return (value - base) / max(255.0 - base, 1.0)
    return (value - base) / max(base, 1.0)


def _features(crop: np.ndarray, base: float, thresh: float) -> np.ndarray:
    """Five numbers describing a square, all relative to its background.

    `base` is this square's background *in this frame* (see `_squares`), and
    every number here is relative to it. That combination -- measured
    background, headroom-relative magnitudes -- is what lets one classifier
    work across board themes and survive last-move highlights.

    The last one carries the colour signal. Mean luminance, the obvious
    choice, is unreliable on ornate piece sets: a *white* queen with heavy
    black line detail averages out surprisingly dark, landing it right next to
    a black piece. Asking instead whether the non-background pixels lean
    bright or dark is robust to that -- the white queen is mostly white with
    thin dark lines whatever shape the designer drew.

    It matters that this is the *balance* of bright to dark rather than two
    separate masses. Raw masses correlate with the occupancy dimensions above,
    so including both effectively triples the weight on "is something here"
    and drowns out "what colour is it" -- measured, that made piece colour
    worse, not better. Normalising to a single ratio keeps the axes
    independent: empty squares sit near zero, white positive, black negative.
    """
    flat = crop.ravel()
    p05, p95 = np.percentile(flat, [5, 95])
    bright = float((flat > base + thresh).mean())
    dark = float((flat < base - thresh).mean())
    return np.array([
        100.0 * _headroom(float(p95), base),    # a white piece pushes this up
        -100.0 * _headroom(float(p05), base),   # a black piece pushes this up
        float(flat.std()),                      # any piece adds internal contrast
        100.0 * _headroom(float(flat.mean()), base),
        50.0 * (bright - dark) / (bright + dark + 1e-3),   # which way it leans
    ], dtype=np.float32)


def _parity(r: int, c: int) -> str:
    return "light" if (r + c) % 2 == 0 else "dark"


# --------------------------------------------------------------------------
# calibration: learn this board's look from a starting position
# --------------------------------------------------------------------------
def looks_like_start(board_img: np.ndarray) -> tuple[bool, str]:
    """Is this really a fresh starting position?

    Worth checking properly, because calibration *trains* on the assumption
    that the outer four ranks are full and the middle four are empty. A board
    one move into a game breaks that assumption quietly: the stray pawn gets
    absorbed into the "empty" class, and the resulting classifier can still
    reproduce a start position on read-back. A false pass there would hand us
    a subtly wrong model, so test the layout directly instead.

    No calibration needed -- a piece adds internal contrast to its square, and
    that separates the two groups on any theme.
    """
    lum = _luma(board_img)
    contrast = {(r, c): float(crop.std()) for r, c, crop, _bg in _squares(lum)}

    middle = [contrast[(r, c)] for r in range(2, 6) for c in range(8)]
    outer = [contrast[(r, c)] for r in (0, 1, 6, 7) for c in range(8)]
    quiet, busy = float(np.median(middle)), float(np.median(outer))
    if busy - quiet < 4.0:
        return False, "cannot tell pieces from empty squares -- is this a chessboard?"

    cut = (quiet + busy) / 2.0
    files = "abcdefgh"
    intruders = [f"r{r+1}c{files[c]}" for r in range(2, 6) for c in range(8)
                 if contrast[(r, c)] > cut]
    missing = [f"r{r+1}c{files[c]}" for r in (0, 1, 6, 7) for c in range(8)
               if contrast[(r, c)] < cut]

    if intruders:
        return False, f"{len(intruders)} piece(s) already out in the middle of the board"
    if missing:
        return False, f"{len(missing)} square(s) missing from the back two ranks"
    return True, "clean starting position"


def _measure(board_img: np.ndarray):
    """Everything about an image that does not depend on the position.

    Both the per-square background (a ring median) and the features built on
    it are position-independent -- only the *labels* depend on which pieces
    are where. Separating the two is what makes searching over candidate
    positions cheap: measure once, then try labellings against it.
    """
    lum = _luma(board_img)
    squares = list(_squares(lum))

    bg_light = [bg for r, c, _crop, bg in squares if _parity(r, c) == "light"]
    bg_dark = [bg for r, c, _crop, bg in squares if _parity(r, c) == "dark"]
    base_light = float(np.median(bg_light))
    base_dark = float(np.median(bg_dark))
    # Scaled to this theme's own contrast, so it means the same thing on a
    # washed-out board as on a high-contrast one.
    thresh = max(12.0, 0.25 * abs(base_light - base_dark))

    feats = np.stack([_features(crop, bg, thresh) for _r, _c, crop, bg in squares])
    parities = [_parity(r, c) for r, c, _crop, _bg in squares]
    return feats, parities, base_light, base_dark, thresh


def _fit(feats: np.ndarray, labels: np.ndarray, parities) -> tuple[dict, np.ndarray]:
    """Centroids per (square colour, class), and the distance normaliser."""
    scale = np.maximum(feats.std(axis=0), 1.0) / FEAT_WEIGHTS
    centroids: dict = {"light": {}, "dark": {}}
    for par in ("light", "dark"):
        for lbl in (EMPTY, WHITE, BLACK):
            sel = [i for i in range(64) if parities[i] == par and labels[i] == lbl]
            if sel:
                centroids[par][str(lbl)] = feats[sel].mean(axis=0).tolist()
    return centroids, scale


def _classify(feats, centroids, scale, parities):
    """-> (predicted labels, weakest margin)."""
    pred = np.zeros(64, dtype=np.int8)
    worst = float("inf")
    for i in range(64):
        cents = centroids[parities[i]]
        f = feats[i] / scale
        d = sorted((float(np.linalg.norm(f - np.asarray(v, np.float32) / scale)), int(l))
                   for l, v in cents.items())
        pred[i] = d[0][1]
        if len(d) > 1:
            worst = min(worst, d[1][0] - d[0][0])
    return pred, (0.0 if worst == float("inf") else worst)


_LEAN = 4   # index of the colour-lean feature: positive is bright, negative dark


def _colours_are_the_right_way_round(centroids: dict) -> bool:
    """Does the class we are calling WHITE actually look lighter than BLACK?"""
    def lean(label):
        vals = [centroids[par][str(label)][_LEAN]
                for par in ("light", "dark") if str(label) in centroids[par]]
        return float(np.mean(vals)) if vals else None

    white, black = lean(WHITE), lean(BLACK)
    if white is None or black is None:
        return True          # nothing to contradict, e.g. one side has no pieces
    return white > black


def _openings(max_plies: int):
    """Positions reachable from the start in at most `max_plies` moves."""
    seen, frontier = {}, [chess.Board()]
    seen[chess.Board().board_fen()] = chess.Board()
    for _ in range(max_plies):
        nxt = []
        for b in frontier:
            for mv in b.legal_moves:
                b.push(mv)
                key = b.board_fen()
                if key not in seen:
                    copy = b.copy()
                    seen[key] = copy
                    nxt.append(copy)
                b.pop()
        frontier = nxt
    return list(seen.values())


def identify_position(board_img: np.ndarray, max_plies: int = 2):
    """Work out which early position is on screen, and train on it.

    Calibration needs known labels, and the starting position is the usual
    source. But by the time you have started a game and switched windows the
    opponent has often already moved, and demanding a pristine board means
    restarting the game -- every time.

    So instead of insisting on one position, try every position within a few
    plies of the start, in both orientations, and keep the labelling that the
    pixels actually support. A wrong labelling trains a muddled classifier
    that cannot reproduce itself, so agreement plus margin separates them
    cleanly.

    -> (learned dict, board, agreement, margin) or None.
    """
    feats, parities, base_light, base_dark, thresh = _measure(board_img)

    best = None
    for pw in (True, False):
        for board in _openings(max_plies):
            labels = render_grid(board, pw).ravel()
            centroids, scale = _fit(feats, labels, parities)
            if not _colours_are_the_right_way_round(centroids):
                # Agreement alone cannot catch this. Swapping WHITE and BLACK
                # throughout is a relabelling the classifier reproduces
                # perfectly -- it simply learns that white pieces look dark --
                # and scores a flawless 64/64 while having the board upside
                # down. Only an absolute cue settles it: white pieces really
                # are the brighter ones.
                continue
            pred, margin = _classify(feats, centroids, scale, parities)
            agree = int((pred == labels).sum())
            key = (agree, margin)
            if best is None or key > best[0]:
                best = (key, pw, board, centroids, scale)

    if best is None:
        return None
    (agree, margin), pw, board, centroids, scale = best
    learned = {
        "centroids": centroids,
        "scale": scale.tolist(),
        "playing_white": bool(pw),
        "base_light": base_light,
        "base_dark": base_dark,
        "thresh": thresh,
        "feat_version": FEAT_VERSION,
    }
    return learned, board, agree, margin


def match_opening(grid: np.ndarray, calib, max_plies: int = 2) -> list:
    """Which early positions look exactly like this already-read grid?

    Used at the start of a game rather than at calibration time: the
    classifier is already trained, so we only need the position. Returns
    every match, and the caller insists on exactly one -- two positions with
    the same pattern of occupied-and-coloured squares are indistinguishable
    to us, and picking either would be a guess.
    """
    return [b for b in _openings(max_plies)
            if np.array_equal(render_grid(b, calib.playing_white), grid)]


def learn_from_start(board_img: np.ndarray) -> dict:
    """Train the 3-class classifier and detect orientation.

    Assumes board_img shows the standard starting position. Which side sits
    at the bottom is worked out here rather than asked for.
    """
    lum = _luma(board_img)
    crops = {(r, c): (crop, bg) for r, c, crop, bg in _squares(lum)}

    # The middle four screen rows are empty at the start whichever way the
    # board is drawn, so they hand us the two base tones. These are kept only
    # to size `thresh` and for diagnostics -- reading uses the per-frame ring.
    base = {"light": [], "dark": []}
    for r in range(2, 6):
        for c in range(8):
            base[_parity(r, c)].append(float(np.median(crops[(r, c)][0])))
    base_light = float(np.median(base["light"]))
    base_dark = float(np.median(base["dark"]))
    # Scaled to this theme's own contrast, so it means the same thing on a
    # washed-out board as on a high-contrast one.
    thresh = max(12.0, 0.25 * abs(base_light - base_dark))

    def feat(r, c):
        crop, bg = crops[(r, c)]
        return _features(crop, bg, thresh)

    top_rows = [(r, c) for r in (0, 1) for c in range(8)]
    bot_rows = [(r, c) for r in (6, 7) for c in range(8)]
    mid = [(r, c) for r in range(2, 6) for c in range(8)]

    # Whichever occupied block is brighter is holding the white pieces.
    top_bright = float(np.mean([feat(r, c)[0] for r, c in top_rows]))
    bot_bright = float(np.mean([feat(r, c)[0] for r, c in bot_rows]))
    playing_white = bot_bright > top_bright
    white_sq = bot_rows if playing_white else top_rows
    black_sq = top_rows if playing_white else bot_rows

    groups = {EMPTY: mid, WHITE: white_sq, BLACK: black_sq}
    all_feats = np.stack([feat(r, c) for r in range(8) for c in range(8)])
    # Dividing the weights out here means read_grid stays a plain normalised
    # nearest-centroid lookup with no weighting logic of its own.
    scale = np.maximum(all_feats.std(axis=0), 1.0) / FEAT_WEIGHTS

    centroids: dict = {"light": {}, "dark": {}}
    for label, squares in groups.items():
        for par in ("light", "dark"):
            sel = [feat(r, c) for r, c in squares if _parity(r, c) == par]
            if sel:
                centroids[par][str(label)] = np.mean(sel, axis=0).tolist()

    # How well separated are the classes? Low numbers mean a risky theme.
    return {
        "centroids": centroids,
        "scale": scale.tolist(),
        "playing_white": bool(playing_white),
        "base_light": base_light,
        "base_dark": base_dark,
        "thresh": thresh,
        "feat_version": FEAT_VERSION,
    }


# --------------------------------------------------------------------------
# reading a live position
# --------------------------------------------------------------------------
def read_grid(board_img: np.ndarray, calib) -> tuple[np.ndarray, float]:
    """-> (8x8 array of EMPTY/WHITE/BLACK, margin of the least-sure square)."""
    lum = _luma(board_img)
    scale = np.asarray(calib.scale, dtype=np.float32)
    grid = np.zeros((8, 8), dtype=np.int8)
    worst = float("inf")

    for r, c, crop, bg in _squares(lum):
        par = _parity(r, c)
        f = _features(crop, bg, calib.thresh) / scale
        dists = sorted(
            (float(np.linalg.norm(f - np.asarray(v, dtype=np.float32) / scale)), int(lbl))
            for lbl, v in calib.centroids[par].items()
        )
        grid[r, c] = dists[0][1]
        if len(dists) > 1:
            worst = min(worst, dists[1][0] - dists[0][0])

    return grid, (0.0 if worst == float("inf") else worst)


def render_grid(board: chess.Board, playing_white: bool) -> np.ndarray:
    """What a given position *should* look like on screen."""
    grid = np.zeros((8, 8), dtype=np.int8)
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(screen_to_square(r, c, playing_white))
            if piece is not None:
                grid[r, c] = WHITE if piece.color == chess.WHITE else BLACK
    return grid


def screen_to_square(row: int, col: int, playing_white: bool) -> int:
    """Screen cell (row 0 = top of screen) -> python-chess square index."""
    if playing_white:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def square_to_screen(sq: int, playing_white: bool) -> tuple[int, int]:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if playing_white:
        return 7 - r, f
    return r, 7 - f


# --------------------------------------------------------------------------
# turning a changed picture into a move
# --------------------------------------------------------------------------
def infer_move(board: chess.Board, observed: np.ndarray, playing_white: bool):
    """Which legal move turns `board` into what we can see?

    Every legal move is scored by how many of the 64 squares it explains, so
    a couple of misread squares cannot derail us. Castling, en passant and
    captures all fall out of this for free.

    Returns (move, score) on success or (None, reason) otherwise.
    """
    if np.array_equal(render_grid(board, playing_white), observed):
        return None, "unchanged"

    scored = []
    for mv in board.legal_moves:
        board.push(mv)
        scored.append((int((render_grid(board, playing_white) == observed).sum()), mv))
        board.pop()
    if not scored:
        return None, "no-legal-moves"

    scored.sort(key=lambda t: -t[0])
    best_score, best_move = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0

    if best_score < MIN_MATCH:
        return None, f"no-match (best {best_score}/64)"

    # Promotions are indistinguishable to this classifier: they differ only
    # in piece type, which we never read. Collapse them and assume a queen.
    tied = [mv for s, mv in scored if s == best_score]
    if len(tied) > 1:
        same_squares = all(
            m.from_square == tied[0].from_square and m.to_square == tied[0].to_square
            for m in tied
        )
        if same_squares:
            queen = next((m for m in tied if m.promotion == chess.QUEEN), tied[0])
            return queen, best_score
        return None, f"ambiguous ({len(tied)} moves tie at {best_score}/64)"

    if best_score - runner_up < MIN_MARGIN:
        return None, f"low margin ({best_score} vs {runner_up})"

    return best_move, best_score


def ascii_grid(grid: np.ndarray) -> str:
    return "\n".join(" ".join(LABELS[int(v)] for v in row) for row in grid)
