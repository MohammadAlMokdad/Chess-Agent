"""The play loop: watch, understand, decide, move.

The whole agent is a small state machine over one invariant -- our in-memory
`chess.Board` always mirrors what is on the screen. Every screen change is
resolved into exactly one legal move before the board advances, so the agent
either understands the position completely or refuses to act.
"""
from __future__ import annotations

import random
import time

import chess
import numpy as np

from . import capture, vision
from .brain import Brain
from .config import (Calibration, POLL_INTERVAL, STABLE_FRAMES, TAKEOVER_PX,
                     MOVE_DELAY)
from .hands import Hands, Aborted, HumanTookOver, check_abort

# Outcomes of one attempt at our own move. "Retry the click" and "I no longer
# know what position we are in" must not be the same answer: retrying from a
# board we know is stale sends a move the site will reject as illegal.
PLAYED, RETRY, LOST = "played", "retry", "lost"


class Agent:
    def __init__(
        self,
        calib: Calibration,
        brain: Brain,
        watch_only: bool = False,
        use_drag: bool = False,
        verbose: bool = True,
        move_delay: tuple[float, float] | None = None,
    ):
        self.calib = calib
        self.brain = brain
        self.hands = Hands(calib, use_drag=use_drag)
        self.watch_only = watch_only
        self.verbose = verbose
        self.move_delay = MOVE_DELAY if move_delay is None else move_delay
        self.board = chess.Board()
        self.our_color = chess.WHITE if calib.playing_white else chess.BLACK
        self._last_grid = None
        self._stable_count = 0

    # -- plumbing ----------------------------------------------------------
    def log(self, *parts) -> None:
        if self.verbose:
            print(*parts, flush=True)

    def _snap(self) -> np.ndarray:
        img = capture.grab_board(self.calib.left, self.calib.top, self.calib.size)
        grid, _ = vision.read_grid(img, self.calib)
        return grid

    def _stable_grid(self, timeout: float = 0.0):
        """Poll until the same reading repeats STABLE_FRAMES times.

        Waiting for stability is what keeps us from reading a board
        mid-animation, when a piece is drawn between two squares.
        """
        deadline = time.time() + timeout if timeout else None
        last, count = None, 0
        while True:
            check_abort()
            grid = self._snap()
            if last is not None and np.array_equal(grid, last):
                count += 1
                if count >= STABLE_FRAMES:
                    return grid
            else:
                last, count = grid, 1
            if deadline and time.time() > deadline:
                return None
            time.sleep(POLL_INTERVAL)

    # -- phases ------------------------------------------------------------
    def wait_for_start(self, timeout: float = 120.0, max_plies: int = 2) -> bool:
        """Block until the screen shows a position we can pick up from.

        Not strictly the starting position. By the time a game is open and
        the window switched, the opponent has usually moved already -- and
        insisting on a pristine board means restarting the game every time.
        Anything within a couple of plies of the start is enough to know
        exactly where we are, so accept that and set the board accordingly.
        """
        self.log("Waiting for a position to pick up from...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            grid = self._stable_grid(timeout=2.0)
            if grid is None:
                continue
            found = vision.match_opening(grid, self.calib, max_plies)
            if len(found) == 1:
                self.board = found[0]
                if self.board.move_stack:
                    self.log(f"  joining after {self._opening_line()}")
                return True
        return False

    def _opening_line(self) -> str:
        replay, out = chess.Board(), []
        for mv in self.board.move_stack:
            out.append(replay.san(mv))
            replay.push(mv)
        return " ".join(f"{i // 2 + 1}.{m}" if i % 2 == 0 else m
                        for i, m in enumerate(out))

    def _paths_to(self, grid: np.ndarray, depth: int) -> list[list[chess.Move]]:
        """Every sequence of at most `depth` legal moves that yields `grid`.

        Shortest first, and it stops descending as soon as a depth explains
        the picture -- a one-move explanation is always preferred to a
        two-move one.
        """
        if np.array_equal(vision.render_grid(self.board, self.calib.playing_white), grid):
            return [[]]
        if depth <= 0:
            return []

        move, _info = vision.infer_move(self.board, grid, self.calib.playing_white)
        if move is not None:
            return [[move]]
        if depth == 1:
            return []

        found = []
        for mv in list(self.board.legal_moves):
            self.board.push(mv)
            for tail in self._paths_to(grid, depth - 1):
                found.append([mv] + tail)
            self.board.pop()
        return found

    def _resync(self, depth: int = 2) -> bool:
        """After an interruption, work out what the board is showing now.

        Two plies, not one. While the agent is paused with its hands off, our
        own half-played move may have landed *and* the opponent may have
        replied, leaving the board two moves ahead of what we last recorded.
        Searching only one move deep makes that look like nonsense and throws
        away a perfectly recoverable game.

        Ambiguity is refused rather than guessed: if several move sequences
        explain the same picture we have no way to pick, and inventing one
        would desync us permanently.
        """
        grid = self._stable_grid(timeout=6.0)
        if grid is None:
            self.log("  !! could not get a steady read of the board")
            return False

        paths = self._paths_to(grid, depth)
        if not paths:
            self.log("  !! board no longer matches the game we were tracking")
            return False
        if len(paths) > 1:
            self.log(f"  !! {len(paths)} different move sequences would explain the "
                     f"board; refusing to guess")
            return False

        for mv in paths[0]:
            self.log(f"  ~~ caught up: {self.board.san(mv)} was played while paused")
            self.board.push(mv)
        return True

    def _linger(self, started_at: float) -> None:
        """Sit on the move until `move_delay` has passed since we started.

        Slept in slices rather than one long sleep so the abort key stays
        responsive -- a kill switch that ignores you for five seconds is not
        a kill switch.
        """
        lo, hi = self.move_delay
        if hi <= 0:
            return
        target = started_at + random.uniform(lo, hi)
        while True:
            check_abort()
            remaining = target - time.time()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def _play_our_move(self) -> str:
        started_at = time.time()
        move, note = self.brain.best_move(self.board)
        san = self.board.san(move)

        if self.watch_only:
            self.log(f"  -> suggest {san}  ({note})   [watch mode: not moving]")
            self.board.push(move)
            return PLAYED

        before = vision.render_grid(self.board, self.calib.playing_white)
        self.log(f"  -> playing {san}  ({note})")
        self._linger(started_at)
        try:
            self.hands.play(move)
        except HumanTookOver:
            # Do not push the move: it may be half-played, fully played, or
            # not played at all. Wait for the mouse back, then look.
            if not self.hands.wait_for_handback(self.log):
                raise Aborted("gave up waiting for the mouse")
            return PLAYED if self._resync() else LOST
        self.board.push(move)
        self.hands.park()

        # Confirm the click actually landed: the board must stop looking the
        # way it did before we moved.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            check_abort()
            if not np.array_equal(self._snap(), before):
                return PLAYED
            time.sleep(POLL_INTERVAL)

        self.board.pop()
        self.log("  !! move did not register on screen; retrying")
        return RETRY

    def _await_opponent(self, stall_timeout: float = 180.0) -> bool:
        """Watch until the opponent's move can be identified, then apply it.

        Gives up after `stall_timeout` of no readable change -- the game may
        have ended, the window may have closed, or we may have lost sync, and
        spinning forever helps in none of those cases.
        """
        mine = vision.render_grid(self.board, self.calib.playing_white)
        complaints = set()
        deadline = time.time() + stall_timeout
        while True:
            check_abort()
            if time.time() > deadline:
                self.log("  !! no readable opponent move; giving up on this game")
                return False

            if self.hands.pointer_drift() > TAKEOVER_PX:
                if not self.hands.wait_for_handback(self.log):
                    raise Aborted("gave up waiting for the mouse")
                if not self._resync():
                    return False
                return True

            grid = self._stable_grid(timeout=1.5)
            if grid is None or np.array_equal(grid, mine):
                time.sleep(POLL_INTERVAL)
                continue

            move, info = vision.infer_move(self.board, grid, self.calib.playing_white)
            if move is not None:
                self.log(f"  <- opponent {self.board.san(move)}  [{info}/64 squares]")
                self.board.push(move)
                return True

            if info != "unchanged" and info not in complaints:
                complaints.add(info)
                self.log(f"  .. board changed but unreadable: {info}")
            time.sleep(POLL_INTERVAL)

    # -- main --------------------------------------------------------------
    def run(self, max_moves: int = 400, stall_timeout: float = 180.0) -> str:
        side = "White" if self.our_color == chess.WHITE else "Black"
        self.log(f"Agent is playing {side} with {self.brain.name}. Press F9 to stop.\n")

        try:
            while not self.board.is_game_over() and self.board.fullmove_number <= max_moves:
                check_abort()
                if self.board.turn == self.our_color:
                    outcome = self._play_our_move()
                    if outcome == RETRY:
                        continue
                    if outcome == LOST:
                        return ("Stopped: lost track of the position after an "
                                "interruption. Recalibrate and restart.")
                elif not self._await_opponent(stall_timeout):
                    return "Stopped: the board stopped making sense (game over, or lost sync)."
        except Aborted:
            return "Stopped by operator (F9)."
        except KeyboardInterrupt:
            return "Stopped (Ctrl-C)."

        if self.board.is_game_over():
            return f"Game over: {self.board.result()} ({_reason(self.board)})"
        return "Move limit reached."


def _reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_repetition():
        return "repetition"
    return "draw"
