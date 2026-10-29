"""Chess Agent -- command line.

    python run.py detect      find the board on screen, print where it is
    python run.py calibrate   learn this board's theme (start position on screen)
    python run.py watch       read the game and print the best move, no mouse
    python run.py play        full auto: read, think, click
    python run.py selftest    offline check of the vision + inference stack
    python run.py fixtures    replay real captures that once broke it
    python run.py loopback    play whole games against a virtual screen
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

import chessagent
from chessagent import capture, vision, board_detect
from chessagent.agent import Agent
from chessagent.brain import Brain
from chessagent.config import (Calibration, CALIB_PATH, DEBUG_DIR, ENGINE_PATH,
                               MIN_CONFIDENCE, MOVE_DELAY)

import os


def _countdown(seconds: int, why: str) -> None:
    print(f"{why}")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  go!      ")


def _find_board(args, retry: float):
    """Scan once, or keep scanning until a convincing board turns up."""
    deadline = time.time() + retry
    attempt = 0
    while True:
        attempt += 1
        hit = board_detect.find_board(min_size=args.min_size)
        if hit is not None and hit[3] >= MIN_CONFIDENCE:
            return hit
        if time.time() >= deadline:
            return hit
        if attempt == 1:
            print(f"No board yet -- watching the screen for up to {retry:.0f}s. "
                  "Open one whenever you are ready.", flush=True)
        time.sleep(2.0)


def cmd_detect(args) -> int:
    _report_coordinates()
    if args.wait:
        _countdown(args.wait, "Bring the board into view.")
    print("Scanning all monitors for a chessboard...", flush=True)
    hit = _find_board(args, args.retry)
    if hit is None:
        print("No board found. Is one visible?")
        return 1
    left, top, size, score = hit
    print(f"Board at ({left}, {top}), {size}x{size} px, square {size / 8:.1f} px")
    print(f"Checkerboard confidence: {score:.2f}  (real boards score ~1.0-3.5)")
    if score < MIN_CONFIDENCE:
        print("\n!! That is too low to be a real board -- this is probably a false hit.")
        print("   Make sure a chessboard is actually visible, then try again.")

    os.makedirs(DEBUG_DIR, exist_ok=True)
    out = os.path.join(DEBUG_DIR, "detected_board.png")
    cv2.imwrite(out, capture.grab_board(left, top, size))
    print(f"Saved what it found -> {out}")
    return 0


def cmd_calibrate(args) -> int:
    if args.wait:
        _countdown(args.wait, "Put a board at its STARTING POSITION on screen.")

    if args.region:
        left, top, size = args.region
        print(f"Using the region you gave: ({left}, {top}) {size}x{size}")
    else:
        hit = _find_board(args, args.retry)
        if hit is None:
            print("Could not find a board. Pass --region LEFT TOP SIZE to set it by hand.")
            return 1
        left, top, size, score = hit
        print(f"Found board at ({left}, {top}), {size}x{size} px  (confidence {score:.2f})")
        if score < MIN_CONFIDENCE and not args.force:
            print("That is too low to be a real board. Is one visible? "
                  "Use --region LEFT TOP SIZE to place it by hand, or --force.")
            return 1

    img = capture.grab_board(left, top, size)

    # Calibration needs known labels. The starting position is the usual
    # source, but the opponent has often moved by the time you get here, so
    # accept anything within a couple of plies and work out which it is.
    found = vision.identify_position(img, max_plies=args.plies)
    if found is None and args.retry:
        print(f"Cannot read this board. Watching for up to {args.retry:.0f}s...", flush=True)
        deadline = time.time() + args.retry
        while time.time() < deadline and found is None:
            time.sleep(2.0)
            img = capture.grab_board(left, top, size)
            found = vision.identify_position(img, max_plies=args.plies)
    if found is None:
        print("\n!! Could not make sense of this board.")
        print("   It must be within a few moves of the start of a game.")
        return 1

    learned, opening, agree, _margin = found
    if agree < 64:
        print(f"\n!! Best reading only explains {agree}/64 squares.")
        if not args.force:
            return 1

    calib = Calibration(left=left, top=top, size=size, **learned)

    # Read the position straight back. If the classifier cannot reproduce the
    # start position it just trained on, nothing downstream will work.
    import chess
    grid, margin = vision.read_grid(img, calib)
    expected = vision.render_grid(opening, calib.playing_white)
    agree = int((grid == expected).sum())

    print(f"\nOrientation: we are {'White' if calib.playing_white else 'Black'} (bottom of board)")
    print(f"Base tones: light {calib.base_light:.0f}, dark {calib.base_dark:.0f}")
    print(f"Read-back: {agree}/64 squares correct, weakest margin {margin:.2f}\n")
    print(vision.ascii_grid(grid))

    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, "calibration.png"), img)

    if agree < 64:
        print("\n!! Could not reproduce the starting position.")
        print("   Check debug/calibration.png -- is it exactly the board, no border?")
        if not args.force:
            return 1

    calib.save()
    print(f"\nSaved -> {CALIB_PATH}")
    return 0


def _report_coordinates() -> bool:
    ok, detail = chessagent.coordinate_check()
    if ok:
        print(f"Screen/mouse coordinates agree: {detail}")
    else:
        print(f"!! Screen and mouse disagree: {detail}")
        print("   Clicks will land in the wrong place. Check display scaling.")
    return ok


def _run(args, watch_only: bool) -> int:
    if not watch_only and not _report_coordinates():
        print("Refusing to drive the mouse with mismatched coordinates.")
        print("Use `watch` to test the vision side safely.")
        return 1
    if not os.path.exists(ENGINE_PATH):
        print(f"Engine missing: {ENGINE_PATH}")
        return 1
    calib = Calibration.load()

    if args.wait:
        _countdown(args.wait, "Switch to the board window.")

    brain = Brain(strength=args.strength, think_time=args.think, depth=args.depth)
    try:
        agent = Agent(calib, brain, watch_only=watch_only, use_drag=args.drag,
                      move_delay=tuple(args.delay))
        if not args.resume and not agent.wait_for_start(timeout=args.start_timeout):
            print("Never saw a starting position. Start a new game, or pass --resume.")
            return 1
        print(agent.run())
        print("\nGame as played:")
        print(" ".join(f"{i // 2 + 1}.{m}" if i % 2 == 0 else m
                       for i, m in enumerate(_san_list(agent.board))))
    finally:
        brain.close()
    return 0


def _san_list(board):
    import chess
    replay, out = chess.Board(), []
    for mv in board.move_stack:
        out.append(replay.san(mv))
        replay.push(mv)
    return out


def cmd_watch(args) -> int:
    return _run(args, watch_only=True)


def cmd_play(args) -> int:
    return _run(args, watch_only=False)


def cmd_selftest(args) -> int:
    from tools.synth import run_selftest
    return run_selftest(games=args.games, verbose=not args.quiet)


def cmd_fixtures(args) -> int:
    from tools.fixtures import run_all
    return run_all()


def cmd_loopback(args) -> int:
    from tools.loopback import run_loopback
    return run_loopback(games=args.games, strength=args.strength,
                        think=args.think, verbose=args.verbose)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--wait", type=int, default=0,
                        help="seconds to wait before starting, to switch windows")
        sp.add_argument("--min-size", type=int, default=240,
                        help="smallest board to look for, in pixels")
        sp.add_argument("--retry", type=float, default=0.0,
                        help="keep scanning this many seconds until a board appears")

    d = sub.add_parser("detect", help="locate a board on screen")
    add_common(d)
    d.set_defaults(func=cmd_detect)

    c = sub.add_parser("calibrate", help="learn the board theme from a start position")
    add_common(c)
    c.add_argument("--region", type=int, nargs=3, metavar=("LEFT", "TOP", "SIZE"),
                   help="skip auto-detection and use this square region")
    c.add_argument("--force", action="store_true", help="save even if read-back fails")
    c.add_argument("--plies", type=int, default=2,
                   help="how many moves into a game it may already be (default 2)")
    c.set_defaults(func=cmd_calibrate)

    for name, fn, helptext in (("watch", cmd_watch, "read and suggest, never click"),
                               ("play", cmd_play, "full auto")):
        s = sub.add_parser(name, help=helptext)
        add_common(s)
        s.add_argument("--strength", type=int, default=None,
                       help="Stockfish skill level 0-20 (default: full strength)")
        s.add_argument("--think", type=float, default=0.5, help="seconds per move")
        s.add_argument("--depth", type=int, default=None, help="fixed depth instead of time")
        s.add_argument("--drag", action="store_true", help="drag pieces instead of click-click")
        s.add_argument("--delay", type=float, nargs=2, metavar=("MIN", "MAX"),
                       default=list(MOVE_DELAY),
                       help="seconds to sit on a move before playing it "
                            f"(default {MOVE_DELAY[0]:g} {MOVE_DELAY[1]:g}; use 0 0 for instant)")
        s.add_argument("--resume", action="store_true",
                       help="do not wait for a starting position (assumes one is showing)")
        s.add_argument("--start-timeout", type=float, default=120.0)
        s.set_defaults(func=fn)

    t = sub.add_parser("selftest", help="offline vision + inference check")
    t.add_argument("--games", type=int, default=20)
    t.add_argument("--quiet", action="store_true")
    t.set_defaults(func=cmd_selftest)

    fx = sub.add_parser("fixtures", help="check real captures that once broke it")
    fx.set_defaults(func=cmd_fixtures)

    lb = sub.add_parser("loopback", help="play full games against a virtual screen")
    lb.add_argument("--games", type=int, default=4)
    lb.add_argument("--strength", type=int, default=None)
    lb.add_argument("--think", type=float, default=0.05)
    lb.add_argument("--verbose", action="store_true", help="print every move")
    lb.set_defaults(func=cmd_loopback)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
