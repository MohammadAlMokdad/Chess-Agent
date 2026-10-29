"""Screen capture. One mss instance, reused, because creating them is slow."""
from __future__ import annotations

import threading

import numpy as np

try:
    import mss
except ImportError:  # pragma: no cover
    raise SystemExit("pip install mss")

_local = threading.local()


def _sct():
    """mss handles are not thread-safe, so keep one per thread."""
    if getattr(_local, "sct", None) is None:
        _local.sct = mss.mss()
    return _local.sct


def virtual_screen() -> dict:
    """Bounding box covering every monitor."""
    return _sct().monitors[0]


def grab(region: dict | None = None) -> np.ndarray:
    """Capture a region as an HxWx3 uint8 BGR array."""
    sct = _sct()
    shot = sct.grab(region or sct.monitors[0])
    return np.asarray(shot, dtype=np.uint8)[:, :, :3]


def grab_board(left: int, top: int, size: int) -> np.ndarray:
    return grab({"left": left, "top": top, "width": size, "height": size})
