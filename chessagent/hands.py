"""Mouse control, plus the kill switch that makes it safe to hand over.

Never run an agent that drives the mouse without a way to stop it that does
not itself require the mouse. Two independent ones here:

  * F9 (or whatever ABORT_KEY is) -- polled globally, works without focus.
  * pyautogui's corner failsafe -- slam the pointer into the top-left of the
    screen and the next call raises.
"""
from __future__ import annotations

import ctypes
import random
import time

import chess
import pyautogui

from .config import (MOVE_DURATION, CLICK_PAUSE, JITTER,
                     TAKEOVER_PX, HANDBACK_QUIET, HANDBACK_TIMEOUT)
from .vision import square_to_screen

pyautogui.FAILSAFE = True   # top-left corner aborts
pyautogui.PAUSE = 0.0       # we do our own, randomised, pacing

ABORT_KEY = 0x78            # VK_F9
_user32 = ctypes.windll.user32


class Aborted(Exception):
    """Raised when the operator asked us to stop."""


class HumanTookOver(Exception):
    """Raised when the pointer moved somewhere the agent did not put it.

    Not an error -- the operator reached for the mouse. The agent abandons
    the move in progress rather than wrestling for the cursor, and waits.
    """


def abort_requested() -> bool:
    """True if the abort key is down right now, focus or not."""
    return bool(_user32.GetAsyncKeyState(ABORT_KEY) & 0x8000)


def check_abort() -> None:
    if abort_requested():
        raise Aborted("abort key pressed")


class Hands:
    """Turns chess squares into clicks on a calibrated board."""

    def __init__(self, calib, use_drag: bool = False):
        self.calib = calib
        self.use_drag = use_drag
        self._aimed: tuple[int, int] | None = None   # where we last put the pointer

    # -- sharing the mouse with a human ------------------------------------
    def pointer_drift(self) -> int:
        """How far the pointer is from where the agent last left it."""
        if self._aimed is None:
            return 0
        x, y = pyautogui.position()
        return max(abs(x - self._aimed[0]), abs(y - self._aimed[1]))

    def check_takeover(self) -> None:
        if self.pointer_drift() > TAKEOVER_PX:
            raise HumanTookOver(f"pointer moved {self.pointer_drift()}px")

    def wait_for_handback(self, log=print) -> bool:
        """Block until the mouse has been still for a while. -> True if handed back."""
        log(f"  ~~ you took the mouse -- pausing until it is still for "
            f"{HANDBACK_QUIET:.0f}s (F9 to stop)")
        deadline = time.time() + HANDBACK_TIMEOUT
        last = pyautogui.position()
        still_since = time.time()
        while time.time() < deadline:
            check_abort()
            time.sleep(0.25)
            now = pyautogui.position()
            if max(abs(now[0] - last[0]), abs(now[1] - last[1])) > 3:
                last, still_since = now, time.time()
                continue
            if time.time() - still_since >= HANDBACK_QUIET:
                self._aimed = None          # forget where we were; re-aim from scratch
                log("  ~~ mouse is idle again, resuming")
                return True
        return False

    def square_point(self, sq: int) -> tuple[int, int]:
        """Screen pixel to aim at for a given chess square."""
        row, col = square_to_screen(sq, self.calib.playing_white)
        cell = self.calib.cell
        cx = self.calib.left + (col + 0.5) * cell
        cy = self.calib.top + (row + 0.5) * cell
        # Land somewhere plausible inside the square rather than dead centre.
        wobble = cell * 0.5 * JITTER
        return (
            int(round(cx + random.uniform(-wobble, wobble))),
            int(round(cy + random.uniform(-wobble, wobble))),
        )

    def _travel(self, point: tuple[int, int]) -> None:
        check_abort()
        self.check_takeover()
        pyautogui.moveTo(*point, duration=random.uniform(*MOVE_DURATION),
                         tween=pyautogui.easeOutQuad)
        self._aimed = point

    def _click(self, point: tuple[int, int]) -> None:
        """Click a specific point, never wherever the cursor happens to be.

        Passing coordinates matters. A bare `click()` fires at the pointer's
        current location, so anything that nudged the mouse since we aimed --
        a human hand, another app warping the cursor -- lands the click on the
        wrong square and plays a move we never chose.
        """
        check_abort()
        self.check_takeover()
        pyautogui.click(point[0], point[1])
        self._aimed = point

    def play(self, move: chess.Move) -> None:
        """Execute a move on the real board.

        Raises HumanTookOver if the pointer wanders mid-move, leaving the
        move half-played rather than finishing it somewhere wrong. The caller
        is expected to verify the board afterwards either way.
        """
        src = self.square_point(move.from_square)
        dst = self.square_point(move.to_square)

        if self.use_drag:
            self._travel(src)
            pyautogui.mouseDown()
            try:
                time.sleep(random.uniform(*CLICK_PAUSE))
                self._travel(dst)
            finally:
                pyautogui.mouseUp()     # never leave a button stuck down
        else:
            # Click-click is more reliable than dragging on both chess.com
            # and lichess, and survives animation better.
            self._click(src)
            time.sleep(random.uniform(*CLICK_PAUSE))
            self._click(dst)

        if move.promotion is not None:
            # Both sites drop the promotion picker over the target square
            # with the queen nearest it, so a second click there takes the
            # queen. Anything else needs a site-specific offset.
            time.sleep(0.35)
            self._click(self.square_point(move.to_square))

    def park(self) -> None:
        """Move the pointer off the board so it cannot obscure a square.

        Doubles as the reference point for takeover detection: between moves
        the pointer should be sitting exactly here, so anything else means a
        human moved it.
        """
        check_abort()
        x = self.calib.left + self.calib.size + 60
        y = self.calib.top + self.calib.size // 2
        try:
            pyautogui.moveTo(x, y, duration=0.15)
            self._aimed = (x, y)
        except Exception:
            pass
