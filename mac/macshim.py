"""macOS compatibility layer. Import this BEFORE anything from chessagent.

Nothing under chessagent/ is modified. The Windows build is the reference
implementation and is left exactly as it is; this module adapts it at
runtime, from the outside, by patching the three places where it assumes
Windows. Keeping the adaptation in one file means a macOS problem can never
break the Windows path.

What actually differs:

1. `chessagent.hands` reads `ctypes.windll.user32` at import time, which does
   not exist off Windows. A stand-in is installed so the import succeeds.

2. The kill switch polls a Windows API for the F9 key. On macOS that is done
   with `pynput` instead, if it is installed.

3. Retina. This is the one that would silently ruin everything: screen
   capture returns physical pixels (2880 wide) while the mouse works in
   logical points (1440 wide), so every click would land at half the
   intended position. Rather than teach the agent about two coordinate
   systems, the conversion is applied at the pyautogui boundary -- the agent
   goes on thinking entirely in capture pixels, as it does on Windows.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if sys.platform == "win32":
    raise SystemExit("This shim is for macOS. On Windows use run.py or play.bat.")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------
# 1. let chessagent.hands import at all
# --------------------------------------------------------------------------
class _FakeUser32:
    """Just enough of user32 for the import to succeed.

    GetAsyncKeyState always reports "not pressed"; the real key detection is
    installed further down. Anything else is a genuine Windows call that
    should never be reached here, so it fails loudly rather than silently.
    """

    def GetAsyncKeyState(self, _vk):     # noqa: N802 - mirroring the Win32 name
        return 0

    def __getattr__(self, name):
        raise AttributeError(f"Windows-only call {name!r} reached on macOS")


class _FakeWindll:
    user32 = _FakeUser32()

    def __getattr__(self, name):
        raise AttributeError(f"Windows-only library {name!r} reached on macOS")


if not hasattr(ctypes, "windll"):
    ctypes.windll = _FakeWindll()


# --------------------------------------------------------------------------
# 2. engine path -- no .exe, and fall back to a Homebrew install
# --------------------------------------------------------------------------
def _runnable(path: str) -> bool:
    """A real file we are actually allowed to execute.

    `os.path.exists` is not enough. An archive whose top-level folder is also
    called "stockfish" can leave a *directory* at this path, and trying to run
    a directory surfaces as a bare "Permission denied" from deep inside
    asyncio -- which says nothing about what is really wrong.
    """
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _find_engine() -> str:
    override = os.environ.get("CHESS_AGENT_ENGINE")
    if override:
        return override
    local = os.path.join(ROOT, "engine", "stockfish")
    if _runnable(local):
        return local
    found = shutil.which("stockfish")      # e.g. brew install stockfish
    return found or local                  # report the local path if unusable


from chessagent import config as _config   # noqa: E402

_config.ENGINE_PATH = _find_engine()


# --------------------------------------------------------------------------
# 3. Retina: convert capture pixels <-> mouse points at the pyautogui edge
# --------------------------------------------------------------------------
import pyautogui                            # noqa: E402

from chessagent import capture              # noqa: E402

_real = {name: getattr(pyautogui, name)
         for name in ("moveTo", "click", "position", "size")}


def _measure_scale() -> float:
    """Capture pixels per mouse point. 2.0 on a Retina display, 1.0 otherwise."""
    try:
        monitor = capture._sct().monitors[1]
        width, _height = _real["size"]()
        return monitor["width"] / float(width) if width else 1.0
    except Exception:
        return 1.0


SCALE = _measure_scale()


def _to_points(x, y):
    if x is not None:
        x = x / SCALE
    if y is not None:
        y = y / SCALE
    return x, y


def _moveTo(x=None, y=None, *args, **kwargs):   # noqa: N802
    x, y = _to_points(x, y)
    return _real["moveTo"](x, y, *args, **kwargs)


def _click(x=None, y=None, *args, **kwargs):
    x, y = _to_points(x, y)
    return _real["click"](x, y, *args, **kwargs)


def _position():
    point = _real["position"]()
    return pyautogui.Point(int(round(point[0] * SCALE)), int(round(point[1] * SCALE)))


def _size():
    # Reported in capture pixels so the agent's coordinate sanity check,
    # which compares this against the captured monitor, still agrees.
    width, height = _real["size"]()
    return pyautogui.Size(int(round(width * SCALE)), int(round(height * SCALE)))


if abs(SCALE - 1.0) > 1e-6:
    pyautogui.moveTo = _moveTo
    pyautogui.click = _click
    pyautogui.position = _position
    pyautogui.size = _size


# --------------------------------------------------------------------------
# 4. the kill switch
# --------------------------------------------------------------------------
from chessagent import hands as _hands       # noqa: E402

KILL_SWITCH = "none"

try:
    from pynput import keyboard as _keyboard

    _held: set = set()

    def _on_press(key):
        _held.add(key)

    def _on_release(key):
        _held.discard(key)

    _listener = _keyboard.Listener(on_press=_on_press, on_release=_on_release)
    _listener.daemon = True
    _listener.start()

    def _abort_requested() -> bool:
        return _keyboard.Key.f9 in _held

    KILL_SWITCH = "F9 (pynput)"
except Exception:                            # pynput missing or not permitted
    def _abort_requested() -> bool:
        return False

    KILL_SWITCH = "none - use Ctrl-C, or shove the pointer into a screen corner"


def _check_abort() -> None:
    if _abort_requested():
        raise _hands.Aborted("abort key pressed")


_hands.abort_requested = _abort_requested
_hands.check_abort = _check_abort


def engine_problem() -> str | None:
    """Why the engine cannot be used, in words, or None if it is fine."""
    path = _config.ENGINE_PATH
    if _runnable(path):
        return None
    if os.path.isdir(path):
        return (f"{path} is a directory, not the engine.\n"
                f"  The archive unpacked into a folder of the same name. Fix with:\n"
                f"    BIN=$(find engine -type f -name 'stockfish*' ! -name '*.gz' | head -1)\n"
                f"    mv \"$BIN\" engine/sf.tmp && rm -rf engine/stockfish && "
                f"mv engine/sf.tmp engine/stockfish\n"
                f"    chmod +x engine/stockfish")
    if os.path.isfile(path):
        return (f"{path} is not executable. Fix with:\n"
                f"    chmod +x engine/stockfish")
    return (f"No engine at {path}.\n"
            f"  Run:  bash mac/setup.command      (or:  brew install stockfish)")


def describe() -> str:
    problem = engine_problem()
    engine = _config.ENGINE_PATH if problem is None else f"{_config.ENGINE_PATH}   [UNUSABLE]"
    text = (f"macOS mode\n"
            f"  engine      : {engine}\n"
            f"  display     : {SCALE:g}x capture-to-pointer scaling"
            f"{' (Retina)' if SCALE > 1 else ''}\n"
            f"  kill switch : {KILL_SWITCH}")
    if problem:
        text += f"\n\n!! Engine problem: {problem}"
    return text
