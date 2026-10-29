# Chess Agent

A screen-reading chess agent. It finds a chessboard on your screen, works out
the position, asks Stockfish for a move, and plays it with the mouse. It works
with any board theme, on any site or app, and needs only one hint: a starting
position.

```
python run.py detect       # find the board on screen
python run.py calibrate    # learn this board's theme (start position showing)
python run.py watch        # read the game, print the best move, never click
python run.py play         # full auto: read, think, click
```

**Press F9 at any time to stop.** Moving the mouse into the top-left corner of
the screen also aborts.

> Using this against human opponents on online chess sites is cheating. See
> [Fair play](#fair-play) before you point it at anything.

---

## Features

- **Finds the board on its own.** Scans every monitor for an 8x8 checker
  pattern, in about 0.6 s.
- **Calibrates itself from one screenshot.** It learns the theme's square
  tones, what pieces look like on them, and which colour you are playing.
  No per-theme constants and no training data.
- **Cannot play an illegal move.** Vision only decides which *legal* move
  explains what changed on screen, so castling, en passant and captures need
  no special handling.
- **Fails safe.** If a frame is ambiguous it refuses and waits for the next
  stable frame. It does not guess.
- **Strong, tunable engine.** Stockfish via UCI, with skill level, think time,
  fixed depth and human-like move delays.
- **Shares the mouse.** If you grab the pointer mid-game, the agent yields
  and resumes once the mouse is idle again.
- **Watch mode.** Reads the game and prints the best move without touching
  the mouse.
- **Offline test harness.** A synthetic chess GUI, real-capture regression
  fixtures, and full games against a virtual screen. No display needed.
- **macOS shim** (experimental, see [`mac/README.md`](mac/README.md)).

## How it works

Three parts, deliberately kept separate.

| Part | Module | What it does |
|---|---|---|
| Eyes | `chessagent/vision.py` | Turns pixels into a position |
| Brain | `chessagent/brain.py` | Stockfish picks the move |
| Hands | `chessagent/hands.py` | Clicks the squares |
| Loop | `chessagent/agent.py` | Keeps it all in sync |

**The brain is not an LLM, on purpose.** A language model plays around 1500
Elo and sometimes suggests illegal moves. Stockfish plays around 3600, answers
in milliseconds, and costs nothing to run. The interesting problems here are
perception and control, not move selection.

**The eyes never identify a piece.** Recognising "that is a black knight" is a
13-class problem that needs training data for every theme and piece set.
Instead each square is classified as **empty**, **white piece** or **black
piece**, and the agent asks:

> *Which legal move would turn the position I already know into the picture I
> can now see?*

Every legal move is scored by how many of the 64 squares it explains, and the
best one wins. This has some useful consequences:

- A wrong move is impossible by construction. The agent only chooses from the
  legal move list.
- Misreads are survivable. A move is accepted at 61 of 64 squares
  (`MIN_MATCH`), and it must beat the runner-up by 2 (`MIN_MARGIN`).
- Failures are safe. With deliberately broken calibration, over about 930
  positions: 73% read correctly, 27% refused, 0% wrong.
- Promotions are the one blind spot. A queen and a knight look the same to a
  3-class classifier, so the agent assumes a queen.

**It calibrates itself.** A starting position is a free labelled training set:
the outer two ranks are occupied and the middle four are empty, whichever way
round the board is drawn. From one screenshot the agent learns the two
empty-square tones, how white and black pieces look against each, and which
colour you play (from which end of the board is brighter). Features are
measured relative to each square's own background, which is read fresh every
frame from a ring around the piece. That is why one classifier handles brown,
green, blue and dark themes, and why last-move highlights do not confuse it.
Calibration reads the position back and refuses to save unless it gets 64/64.

**Finding the board.** `board_detect.py` splits candidate regions into an 8x8
grid and scores how cleanly the 64 cell means separate into two colour
clusters. A coarse pass on a downscaled screenshot narrows it down, then a
full-resolution pass finds the exact pixel. Real boards score about 1.0-3.5;
a desktop with no board scores about 0.2, so anything under 0.7 is rejected.

## Tech stack

- Python
- [python-chess](https://github.com/niklasf/python-chess) for rules, legal
  moves and the UCI engine protocol
- [Stockfish](https://stockfishchess.org/) as the engine (any UCI engine works)
- [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/) for vision
- [mss](https://github.com/BoboTiG/python-mss) for screen capture
- [PyAutoGUI](https://github.com/asweigart/pyautogui) (with Pillow) for mouse
  control
- Win32 APIs via `ctypes` for DPI awareness and the global F9 key

## Prerequisites

- **Windows 10 or 11.** Windows is the reference platform. macOS is supported
  through a shim in [`mac/`](mac/README.md) that has not been tested on real
  hardware. Linux is not supported.
- **Python 3.9 or newer** (developed on 3.14).
- **Stockfish**, downloaded separately. It is not included in this repository.

## Installation

### Windows

```
git clone https://github.com/MohammadAlMokdad/Chess-Agent.git
cd Chess-Agent
python -m pip install -r requirements.txt
```

Then install Stockfish:

1. Download the Windows build from
   [stockfishchess.org/download](https://stockfishchess.org/download/) or the
   [GitHub releases](https://github.com/official-stockfish/Stockfish/releases).
   The AVX2 build suits most modern CPUs; the plain x86-64 build runs anywhere.
2. Unzip it, then copy the `.exe` into an `engine` folder in the project root
   and rename it to `stockfish.exe`:

   ```
   mkdir engine
   copy path\to\stockfish-windows-x86-64-avx2.exe engine\stockfish.exe
   ```

The agent expects the engine at `engine/stockfish.exe`. To use another path or
another UCI engine, change `ENGINE_PATH` in `chessagent/config.py`.

Check that everything loads (this needs no screen or engine):

```
python run.py selftest
```

### macOS (experimental)

```
bash mac/setup.command
```

This installs the Python packages plus `pyobjc` and `pynput`, downloads the
Stockfish 19 universal macOS build into `engine/stockfish`, and runs a short
self-test. Alternatively run `brew install stockfish`; the shim falls back to a
`stockfish` on your `PATH`. You also need to grant your terminal **Screen
Recording** and **Accessibility** permissions. See
[`mac/README.md`](mac/README.md) for details.

## Usage

### One click (Windows)

1. Start a **new** game on your chess site, with the board fully visible.
2. Double-click `play.bat`, or run it from a terminal.
3. Don't touch the mouse.

`play.bat` runs `calibrate --retry 400` (it waits up to 400 s for a board),
then `play --think 0.4`. Any extra arguments are passed to `play`:

```
play.bat --strength 6
play.bat --delay 1 2
```

On macOS, `bash mac/play.command` does the same.

### Commands

| Command | What it does |
|---|---|
| `python run.py detect` | Find the board, print its position and confidence, save `debug/detected_board.png` |
| `python run.py calibrate` | Learn the theme from a starting position, write `calibration.json` |
| `python run.py watch` | Read the game and print the engine's move. Never clicks |
| `python run.py play` | Full auto: read, think, click |
| `python run.py selftest` | Offline check of vision and move inference on synthetic boards |
| `python run.py fixtures` | Replay real captures that once broke the reader |
| `python run.py loopback` | Play whole games against a virtual screen |

On macOS, use `python3 mac/run_mac.py` in place of `python run.py`.

### Options

Shared by `detect`, `calibrate`, `watch` and `play`:

| Flag | Default | Meaning |
|---|---|---|
| `--wait N` | 0 | Count down N seconds first, so you can switch windows |
| `--retry SECONDS` | 0 | Keep scanning this long until a board appears |
| `--min-size PX` | 240 | Smallest board to look for |

`calibrate` only:

| Flag | Default | Meaning |
|---|---|---|
| `--region LEFT TOP SIZE` | | Skip detection and use this square screen region |
| `--plies N` | 2 | How many moves into a game the board may already be |
| `--force` | | Save even if the read-back is not 64/64 |

`watch` and `play`:

| Flag | Default | Meaning |
|---|---|---|
| `--strength 0-20` | full | Stockfish skill level |
| `--think SECONDS` | 0.5 | Engine time per move (varied slightly each move) |
| `--depth N` | | Search to a fixed depth instead of a time limit |
| `--delay MIN MAX` | 3 5 | Total seconds from seeing the opponent's move to clicking. `0 0` plays instantly |
| `--drag` | | Drag pieces instead of click-click |
| `--resume` | | Start mid-game instead of waiting for a starting position |
| `--start-timeout SECONDS` | 120 | How long to wait for a starting position |

Examples:

```
python run.py play --strength 6 --think 0.3   # weakened, for a fair game
python run.py play --delay 0 0                # move the instant it decides
python run.py play --wait 5                   # 5 s to switch windows first
python run.py calibrate --region 420 180 640  # set the board rectangle by hand
```

### Stopping and taking over

- **F9** stops the agent at any time, whichever window has focus.
- Moving the pointer into the **top-left corner** triggers PyAutoGUI's
  failsafe.
- If you move the mouse more than 20 px from where the agent left it, the
  agent assumes you took over. It abandons the move in progress and resumes
  once the mouse has been still for 2.5 s.

## Configuration and calibration

### calibration.json

`python run.py calibrate` writes `calibration.json` to the project root. It
holds the board's screen rectangle (`left`, `top`, `size` in pixels), your
colour, the theme's base tones, and the learned classifier centroids. It only
makes sense for one monitor layout, board position and theme, so it is
git-ignored and every user creates their own.

Recalibrate when:

- the board window moves or is resized,
- you change board theme, piece set or site, or
- the vision code changes (`FEAT_VERSION` in `vision.py`). The agent refuses
  to load a stale calibration and tells you to rerun it.

Calibration needs a starting position, or a game at most `--plies` moves in.
If it fails, look at `debug/calibration.png`: it should show exactly the board,
with no border.

### Display scaling

On start-up the agent declares itself DPI-aware, so screen capture and mouse
clicks share one coordinate system on scaled displays (125%, 150%). `play`
checks this and refuses to drive the mouse if capture and mouse resolutions
disagree. `watch` still works in that case.

### Tuning

All in `chessagent/config.py`:

| Setting | Meaning |
|---|---|
| `ENGINE_PATH` | Where the UCI engine lives (default `engine/stockfish.exe`) |
| `INNER_CROP` | How much of each square to sample. Lower it if borders or coordinate labels bleed in |
| `STABLE_FRAMES` | Identical reads needed before trusting a change. Raise it for slow piece animations |
| `MIN_MATCH` / `MIN_MARGIN` | How certain the agent must be before accepting a move. Raising them makes it refuse more and misread less |
| `MOVE_DELAY` | Default for `--delay`. The total gap from seeing the opponent's move to clicking, engine time included, so `--think` does not change the pacing |
| `MOVE_DURATION`, `CLICK_PAUSE`, `JITTER` | Mouse travel time, pause between clicks, and aim scatter |
| `TAKEOVER_PX`, `HANDBACK_QUIET` | How far the pointer must stray before the agent yields, and how long it must be still before it resumes |

### Environment variables

The app reads no `.env` file. Two optional variables exist:

| Variable | Used by | Meaning |
|---|---|---|
| `CHESS_AGENT_ENGINE` | macOS shim | Path to the engine binary, overriding `engine/stockfish` and `PATH` |
| `FIXTURE_VERBOSE` | `run.py fixtures` | Set to any value to print every check, not only failures |

## Testing

There is no pytest suite. The tests are built into the CLI:

```
python run.py selftest     # synthetic boards; no screen, no engine
python run.py fixtures     # real screen captures; no screen, no engine
python run.py loopback     # full games against a virtual screen; needs Stockfish
```

- **selftest** renders positions with a synthetic chess GUI
  (`tools/synth.py`): every theme in that file, both orientations, and three
  square sizes (48, 72 and 96 px) with sensor noise and last-move highlights.
  It checks every inferred move against the move actually played. Options:
  `--games N` (default 20), `--quiet`. The last recorded run, on five themes,
  inferred 2213/2213 moves correctly with 0 square misreads out of 141,632.
- **fixtures** replays real captures from `tests/fixtures/`. Each fixture is a
  start frame for calibration, the frame that caused trouble, and a
  `case.json` saying what it should read as. Every bug found in a live game
  gets its frame saved here.
- **loopback** runs the real agent (vision, Stockfish and the state machine)
  against an in-memory board, with clicks applied to that board instead of to
  the OS. Options: `--games N` (default 4), `--strength`, `--think` (default
  0.05), `--verbose`.

## Results

Two live games, both won by checkmate playing White against the site's
computer opponent. Every opponent move in both games was read at 64/64
squares, and moving between the sites took only `detect` then `calibrate`,
with no code changes.

| | lichess (brown, cburnett, 712 px) | chess.com (green, neo, 1080 px) |
|---|---|---|
| Worst square margin | 0.34 | 1.04 |
| Ambiguous frames | 2 | 0 |

The two ambiguous lichess frames came straight after the agent's own move,
catching the board mid-animation. The agent declined to guess and read the
real move on the next stable frame.

<details>
<summary>Game scores</summary>

lichess:

```
1.e4 e5 2.Nf3 Nf6 3.Nxe5 Qe7 4.d4 Nxe4 5.Bd3 d5 6.O-O Nc6 7.Bb5 a6
8.Bxc6+ bxc6 9.Re1 Qe6 10.f3 f6 11.Nd3 h6 12.fxe4 dxe4 13.Qh5+ Ke7
14.Nc5 Qg4 15.Rxe4+ Qxe4 16.Nxe4 Rb8 17.Nbc3 Rb4 18.Bf4 Bd7 19.Bxc7 a5
20.Nd6 g6 21.Qxg6 Rh7 22.Re1+ Be6 23.Qe8#
```

chess.com:

```
1.e4 e6 2.d4 Qe7 3.Nc3 c6 4.Bd3 d5 5.Nf3 dxe4 6.Nxe4 b5 7.Ne5 Bb7 8.Bg5 a6
9.Bxe7 Nxe7 10.Nd6+ Kd8 11.Ndxf7+ Kc7 12.Nxh8 Kb6 13.c4 Ka7 14.O-O g6
15.Qc1 h5 16.cxb5 axb5 17.Qc5+ Ka6 18.a4 Bc8 19.Nhf7 Ra7 20.axb5+ Kb7 21.Qxa7#
```

</details>

### The bug those games did not catch

A third game deadlocked. Black played f7-f6; f6 is a dark square, and
chess.com's yellow last-move highlight lifted it from 133 to 185 luma,
*brighter than the grey black pawn standing on it*. Measured against a
background tone stored at calibration time, it read as a white piece. The
true move scored 63/64, `MIN_MARGIN` rejected it, and the agent waited forever
for a move it had already refused. It never played a wrong move, but a
permanent stall is still a failure. Two fixes:

1. **Measure the background, don't assume it.** Each square's empty tone is
   now read every frame from a ring around the piece (`_squares`). The ring is
   highlighted too, so it tracks.
2. **Use relative magnitudes.** A highlight moves the background but not the
   piece. Magnitudes are now fractions of the available *headroom*
   (`_headroom`), which barely moves: 93% unhighlighted versus 90%
   highlighted.

That frame is kept in `tests/fixtures/chesscom-neo/` and checked by
`run.py fixtures`. Its weakest square margin went from 0.36 to 1.97. The
rematch was won by checkmate in 23 moves, including fifteen consecutive
checks, with every opponent move read at 64/64.

## Limits

- Calibration needs a starting position (or a game a couple of moves in). A
  new theme means recalibrating.
- Moving or resizing the board window invalidates calibration.
- Underpromotion is read as a queen promotion, and promotions are played by
  clicking the queen in the site's picker.
- The board must stay unobstructed. The agent parks the pointer beside the
  board after each move for this reason.

## Project structure

```
run.py              CLI entry point (all commands above)
play.bat            Windows one-click launcher: calibrate, then play
requirements.txt    Python dependencies
chessagent/
  __init__.py       DPI awareness, screen/mouse coordinate check
  config.py         paths, tunables, Calibration dataclass
  board_detect.py   finds the board on any monitor
  capture.py        screen capture (mss)
  vision.py         square classifier, self-calibration, move inference
  brain.py          Stockfish wrapper
  hands.py          mouse control, F9 kill switch, takeover detection
  agent.py          the watch, think, move loop
tools/
  synth.py          synthetic chess GUI and selftest
  fixtures.py       regression checks on real captures
  loopback.py       end-to-end games against a virtual screen
tests/fixtures/     real screen captures used by `run.py fixtures`
mac/                macOS shim, launchers and setup script
engine/             not in the repo; put the Stockfish binary here
```

Generated locally and git-ignored: `calibration.json`, `debug/` (screenshots
from `detect` and `calibrate`) and `engine/`.

## Fair play

Running this against human opponents on chess.com or lichess is cheating. It
breaks their terms of service and gets accounts permanently closed. Their
detection is good, and moving the mouse like a person does not defeat it.
Against the sites' computer opponents, a local engine or an offline GUI, none
of that applies.

Lichess also has an official [Bot API](https://lichess.org/api#tag/Bot) where
engine play is explicitly allowed. If you want a bot that plays strong chess,
rather than one that drives a mouse, that route is legitimate and far more
reliable than screen-scraping.

## Credits and licenses

- **[Stockfish](https://stockfishchess.org/)** is licensed under the GNU GPL
  v3. It is not distributed with this repository; you download it separately.
- **[python-chess](https://github.com/niklasf/python-chess)**: GPL v3.
- **mss** (MIT), **PyAutoGUI** (BSD 3-Clause), **NumPy** (BSD 3-Clause),
  **OpenCV** (Apache 2.0), **Pillow** (MIT-CMU).
- macOS extras: **pyobjc** (MIT), **pynput** (LGPL v3).
- The images in `tests/fixtures/` are screenshots of a chess.com board using
  its "neo" piece set. That artwork belongs to Chess.com.

This project does not have a license yet. Until one is added, all rights are
reserved by default.
