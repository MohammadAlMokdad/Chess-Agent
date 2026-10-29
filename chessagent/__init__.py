"""Chess Agent.

DPI awareness is declared here, at package import, before anything touches
the screen. It matters more than it looks: on a scaled display (125%, 150%)
a DPI-unaware process sees a virtualised 1707x1067 desktop while screen
capture returns real 2560x1600 pixels. Mouse clicks would then land about a
third of the way off -- consistently, and only on scaled machines, which is
a miserable bug to chase.

Setting it once, first, means capture and mouse control share one coordinate
system. Must happen before mss or pyautogui do any work; Windows ignores
later attempts to change it.
"""
from __future__ import annotations

import ctypes
import sys


def _declare_dpi_aware() -> str:
    if not sys.platform.startswith("win"):
        return "n/a"
    # Per-monitor v2 is the good one: correct on mixed-scaling multi-monitor
    # setups. Fall back through the older APIs on older Windows.
    try:
        PER_MONITOR_V2 = ctypes.c_void_p(-4)
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(PER_MONITOR_V2):
            return "per-monitor-v2"
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return "system"
    except Exception:
        return "unaware"


DPI_MODE = _declare_dpi_aware()


def coordinate_check() -> tuple[bool, str]:
    """Confirm screen capture and the mouse agree on where pixels are."""
    import pyautogui

    from . import capture

    primary = capture._sct().monitors[1]
    mouse_w, mouse_h = pyautogui.size()
    ok = (primary["width"], primary["height"]) == (mouse_w, mouse_h)
    detail = (f"capture {primary['width']}x{primary['height']} vs "
              f"mouse {mouse_w}x{mouse_h} (dpi: {DPI_MODE})")
    return ok, detail
