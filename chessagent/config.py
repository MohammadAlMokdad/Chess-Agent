"""Paths, tunables, and on-disk calibration state."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_PATH = os.path.join(ROOT, "engine", "stockfish.exe")
CALIB_PATH = os.path.join(ROOT, "calibration.json")
DEBUG_DIR = os.path.join(ROOT, "debug")

# --- vision tunables -------------------------------------------------------
INNER_CROP = 0.62      # fraction of a square sampled, centred (skips gridlines/coords)

# The background tone of each square is measured every frame from a ring
# around the piece, rather than assumed from calibration. Last-move
# highlights change a square's background, and by a wildly different amount
# depending on its colour -- on a real chess.com board the same highlight
# lifts a dark square by +52 luma and a light one by +1. The ring is
# highlighted too, so it tracks that automatically.
RING_INNER = 0.76      # inside edge of the ring: outside the piece
RING_OUTER = 0.97      # outside edge: inside the square, off the gridlines
STABLE_FRAMES = 3      # identical reads required before trusting a board change
POLL_INTERVAL = 0.12   # seconds between screen polls

# Real boards score ~1.0-3.5 on the checkerboard test; an ordinary desktop
# with no board on it scores ~0.2. Below this we assume nothing was found.
MIN_CONFIDENCE = 0.7

# --- move-matching tunables ------------------------------------------------
MIN_MATCH = 61         # of 64 squares, for a legal move to be accepted
MIN_MARGIN = 2         # winning move must beat runner-up by this many squares

# --- hands -----------------------------------------------------------------
MOVE_DURATION = (0.09, 0.22)   # mouse travel time range, seconds
CLICK_PAUSE = (0.05, 0.13)     # pause between the two clicks of a move
JITTER = 0.22                  # click offset, as a fraction of half a square

# How long to sit on a move before playing it. This is the *total* gap
# between seeing the opponent's move and clicking, engine thinking included,
# so changing how long the engine thinks does not change the pacing.
MOVE_DELAY = (3.0, 5.0)

# If the pointer is further than this from where the agent last put it, a
# human has grabbed the mouse. Fighting them is the wrong answer: the agent
# yields and waits instead, like cruise control letting go when you brake.
TAKEOVER_PX = 20
HANDBACK_QUIET = 2.5           # seconds of stillness before taking over again
HANDBACK_TIMEOUT = 600.0       # give up waiting after this long

EMPTY, WHITE, BLACK = 0, 1, 2
LABELS = {EMPTY: ".", WHITE: "W", BLACK: "b"}


@dataclass
class Calibration:
    """Everything learned from one look at a starting position."""
    left: int
    top: int
    size: int                       # board is square: size x size pixels
    playing_white: bool             # True if our pieces are on the bottom
    base_light: float = 0.0         # empty light-square tone
    base_dark: float = 0.0          # empty dark-square tone
    thresh: float = 12.0            # bright/dark mass cutoff, scaled to the theme
    feat_version: int = 0           # which feature vector this was trained with
    centroids: dict = field(default_factory=dict)   # "light"/"dark" -> {label -> feature vec}
    scale: list = field(default_factory=list)       # per-dimension normaliser

    @property
    def cell(self) -> float:
        return self.size / 8.0

    def save(self, path: str = CALIB_PATH) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str = CALIB_PATH) -> "Calibration":
        if not os.path.exists(path):
            raise FileNotFoundError(
                "No calibration.json found. Run:  python run.py calibrate"
            )
        with open(path, encoding="utf-8") as fh:
            calib = cls(**json.load(fh))
        # A calibration trained on an older feature vector would be read with
        # the wrong number of dimensions -- wrong quietly, which is worse than
        # not loading at all.
        from .vision import FEAT_VERSION
        if calib.feat_version != FEAT_VERSION:
            raise SystemExit(
                f"calibration.json was made by an older version of the vision code "
                f"(v{calib.feat_version}, now v{FEAT_VERSION}).\n"
                f"Run:  python run.py calibrate"
            )
        return calib
