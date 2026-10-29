# Running on macOS

The Windows build in the project root is the reference implementation and is
**not modified by anything in this folder**. `macshim.py` adapts it at
runtime instead, so a macOS problem can never break the Windows path.

## Setup, once

```
bash mac/setup.command
```

Or double-click `setup.command` in Finder. It installs the Python packages,
downloads the universal macOS Stockfish build into `engine/stockfish`, and
runs a quick self-test.

### Then grant two permissions

**System Settings → Privacy & Security**, and add your terminal app
(Terminal, iTerm, or VS Code if you run it from there) to:

| Permission | Without it |
|---|---|
| **Screen Recording** | Screenshots come back blank or show only the desktop wallpaper — the agent will report that it cannot find a board |
| **Accessibility** | The mouse never moves; clicks are silently dropped |

**Quit and reopen the terminal afterwards.** macOS only applies these on a
fresh launch, and this catches almost everyone.

## Playing

```
bash mac/play.command
```

Or double-click `play.command`. Extra options pass straight through:

```
bash mac/play.command --strength 6
bash mac/play.command --delay 1 2
```

The individual commands, if you prefer them:

```
python3 mac/run_mac.py detect
python3 mac/run_mac.py calibrate
python3 mac/run_mac.py play
python3 mac/run_mac.py selftest
```

Every command from the main README works, with `python3 mac/run_mac.py`
substituted for `python run.py`.

## What the shim actually changes

Three things, and nothing else.

**1. It lets the code import.** `chessagent/hands.py` reads
`ctypes.windll.user32` at import time. A stand-in is installed so that line
succeeds.

**2. The kill switch.** F9 is polled through a Windows API. On macOS that is
done with `pynput`. If `pynput` is missing or has not been granted
Accessibility permission, **F9 will not work** — the shim says so at startup,
and Ctrl-C in the terminal, or shoving the pointer into a screen corner,
still stop it.

**3. Retina scaling.** This is the one that would otherwise ruin everything
silently. Screen capture returns physical pixels (2880 wide on a 1440-point
display) while the mouse works in logical points, so every click would land
at half the intended position. The conversion is applied at the pyautogui
boundary, so the agent goes on thinking purely in capture pixels exactly as
it does on Windows. Verified by test: a request for capture pixel
`(2000, 1000)` reaches the OS as point `(1000, 500)`.

The startup banner reports all three, so you can see what it decided:

```
macOS mode
  engine      : /path/to/engine/stockfish
  display     : 2x capture-to-pointer scaling (Retina)
  kill switch : F9 (pynput)
```

## Untested

I wrote and verified this on Windows. The coordinate maths, import order and
engine-path resolution are covered by tests; **the shim has never been run on
an actual Mac.** Expect to hit something on the first run — the permissions
above are the most likely culprit, followed by `pyobjc` (pyautogui cannot
import on macOS without it, which the setup script installs).

If `detect` reports no board while one is plainly on screen, that is almost
always Screen Recording permission, not the detector.
