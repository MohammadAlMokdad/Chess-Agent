"""Regression tests against real screen captures.

The synthetic renderer in `synth.py` is good for breadth -- many themes,
scales and positions -- but it only ever contains the failure modes I thought
to model. Every bug found in a live game gets its frame saved here instead,
so the exact pixels that broke it are checked forever after.

Each fixture is a directory holding:
    start.png        a starting position, used to calibrate
    highlighted.png  the frame that caused trouble
    case.json        what the frame is, and what it should read as
"""
from __future__ import annotations

import json
import os

import chess
import cv2
import numpy as np

from chessagent import vision
from chessagent.config import Calibration, LABELS

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tests", "fixtures")

_NAME_TO_LABEL = {"EMPTY": 0, "WHITE": 1, "BLACK": 2}


def _square_to_rc(name: str, playing_white: bool) -> tuple[int, int]:
    return vision.square_to_screen(chess.parse_square(name), playing_white)


def run_fixture(path: str, verbose: bool = True) -> tuple[int, int]:
    """-> (failures, checks) for one fixture directory."""
    with open(os.path.join(path, "case.json"), encoding="utf-8") as fh:
        case = json.load(fh)

    start = cv2.imread(os.path.join(path, "start.png"))
    frame = cv2.imread(os.path.join(path, "highlighted.png"))
    if start is None or frame is None:
        raise SystemExit(f"fixture images missing in {path}")

    fails = checks = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal fails, checks
        checks += 1
        if not ok:
            fails += 1
        if verbose and (not ok or os.environ.get("FIXTURE_VERBOSE")):
            print(f"    {'ok  ' if ok else 'FAIL'} {label}{(' -- ' + detail) if detail else ''}")

    # Calibration must work on the real start frame.
    ok_start, why = vision.looks_like_start(start)
    check(ok_start, "start.png recognised as a starting position", why)
    calib = Calibration(left=0, top=0, size=start.shape[0], **vision.learn_from_start(start))
    check(calib.playing_white == case["playing_white"], "orientation")

    grid, _margin = vision.read_grid(start, calib)
    expected = vision.render_grid(chess.Board(), calib.playing_white)
    agree = int((grid == expected).sum())
    check(agree == 64, "start position reads back 64/64", f"got {agree}/64")

    # The frame that actually broke it.
    live, margin = vision.read_grid(frame, calib)
    for sq_name, want in case.get("expect_square", {}).items():
        r, c = _square_to_rc(sq_name, calib.playing_white)
        got = int(live[r, c])
        check(got == _NAME_TO_LABEL[want], f"square {sq_name} reads {want}",
              f"read {LABELS[got]}")

    if case.get("expect_fen_board"):
        want_grid = vision.render_grid(chess.Board(case["expect_fen_board"] + " w - - 0 1"),
                                       calib.playing_white)
        wrong = int((live != want_grid).sum())
        check(wrong == 0, "whole frame reads correctly", f"{wrong} square(s) wrong")

    if verbose:
        print(f"  {os.path.basename(path):24} {checks - fails}/{checks} checks, "
              f"weakest margin {margin:.2f}")
    return fails, checks


def run_all(verbose: bool = True) -> int:
    if not os.path.isdir(ROOT):
        print("no fixtures directory")
        return 0
    dirs = sorted(os.path.join(ROOT, d) for d in os.listdir(ROOT)
                  if os.path.isdir(os.path.join(ROOT, d)))
    if not dirs:
        print("no fixtures")
        return 0
    print(f"Real-capture fixtures ({len(dirs)}):")
    total_f = total_c = 0
    for d in dirs:
        f, c = run_fixture(d, verbose)
        total_f += f
        total_c += c
    print(f"\n{total_c - total_f}/{total_c} checks passed")
    return 0 if total_f == 0 else 1
