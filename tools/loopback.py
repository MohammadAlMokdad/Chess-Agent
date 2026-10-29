"""End-to-end test: the real Agent plays a real game against a virtual screen.

Everything runs for real -- board reading, move inference, Stockfish, the
state machine -- except that screen capture is served from a rendered image
and mouse clicks are applied to that image instead of to Windows. If this
passes, the only untested link left is whether the pointer lands where the
calibration says it should.
"""
from __future__ import annotations

import random
import time

import chess
import numpy as np

from chessagent import capture, vision, agent as agent_mod, hands as hands_mod
from chessagent.hands import HumanTookOver
from chessagent.agent import Agent
from chessagent.brain import Brain
from chessagent.config import Calibration
from tools.synth import render


class VirtualScreen:
    """A chess site that exists only in memory."""

    def __init__(self, theme: str, playing_white: bool, cell: int = 72,
                 reply_delay: float = 0.25, seed: int = 0):
        self.board = chess.Board()
        self.theme, self.playing_white, self.cell = theme, playing_white, cell
        self.reply_delay = reply_delay
        self.rng = random.Random(seed)
        self.last = None
        self.rejected = []
        # If the agent has Black, the site opens -- after a beat, so the agent
        # still gets to see a clean starting position first.
        self._pending_at = None if playing_white else time.time() + 1.5

    def image(self) -> np.ndarray:
        # The opponent "thinks" for a moment before its reply appears.
        if self._pending_at is not None and time.time() >= self._pending_at:
            self._pending_at = None
            if not self.board.is_game_over():
                mv = self.rng.choice(list(self.board.legal_moves))
                self.board.push(mv)
                self.last = mv
        hl = (self.last.from_square, self.last.to_square) if self.last else None
        return render(self.board, self.playing_white, self.theme, self.cell,
                      highlight=hl, noise=2.0)

    def click_move(self, move: chess.Move) -> None:
        """What the site does when the agent plays a move."""
        if move not in self.board.legal_moves:
            self.rejected.append(move)
            return
        self.board.push(move)
        self.last = move
        self._pending_at = time.time() + self.reply_delay


def play_one(theme: str, playing_white: bool, strength: int, think: float,
             seed: int, verbose: bool, interrupt_every: int = 0,
             interrupt_lands: bool = True):
    screen = VirtualScreen(theme, playing_white, seed=seed)
    calib = Calibration(left=0, top=0, size=screen.cell * 8,
                        **vision.learn_from_start(
                            render(chess.Board(), playing_white, theme, screen.cell)))
    assert calib.playing_white == playing_white, "orientation misread"

    orig_grab, orig_play = capture.grab_board, hands_mod.Hands.play
    orig_wait = hands_mod.Hands.wait_for_handback
    capture.grab_board = lambda *a, **k: screen.image()
    agent_mod.capture.grab_board = capture.grab_board

    plays = {"n": 0, "interrupts": 0}

    def fake_play(self, move):
        plays["n"] += 1
        # Simulate the operator grabbing the mouse mid-move. Two flavours:
        # the click had already landed, or it had not.
        if interrupt_every and plays["n"] % interrupt_every == 0:
            plays["interrupts"] += 1
            if interrupt_lands:
                screen.click_move(move)
            raise HumanTookOver("simulated takeover")
        screen.click_move(move)

    hands_mod.Hands.play = fake_play
    hands_mod.Hands.wait_for_handback = lambda self, log=print: True

    brain = Brain(strength=strength, think_time=think)
    try:
        # No lingering in tests: pacing is a cosmetic concern and would
        # turn a two-minute suite into a two-hour one.
        ag = Agent(calib, brain, verbose=verbose, move_delay=(0.0, 0.0))
        ag.hands.park = lambda: None
        assert ag.wait_for_start(timeout=10), "never saw start position"
        outcome = ag.run(max_moves=200, stall_timeout=20.0)
    finally:
        brain.close()
        capture.grab_board = orig_grab
        agent_mod.capture.grab_board = orig_grab
        hands_mod.Hands.play = orig_play
        hands_mod.Hands.wait_for_handback = orig_wait

    # The agent's internal board must match the site's, move for move.
    synced = ag.board.board_fen() == screen.board.board_fen()
    return {
        "outcome": outcome,
        "plies": len(ag.board.move_stack),
        "synced": synced,
        "rejected": len(screen.rejected),
        "agent_fen": ag.board.board_fen(),
        "site_fen": screen.board.board_fen(),
        "interrupts": plays["interrupts"],
    }


def run_loopback(games: int = 4, strength: int | None = None, think: float = 0.08,
                 verbose: bool = False) -> int:
    themes = ["lichess-brown", "chesscom-green", "dark", "low-contrast"]
    failures = 0
    print(f"{'theme':16} {'side':5} {'plies':>6} {'illegal':>8} {'sync':>6}  outcome")
    print("-" * 78)
    for i in range(games):
        theme = themes[i % len(themes)]
        white = (i % 2 == 0)
        r = play_one(theme, white, strength, think, seed=i, verbose=verbose)
        bad = (not r["synced"]) or r["rejected"]
        failures += bool(bad)
        print(f"{theme:16} {'White' if white else 'Black':5} {r['plies']:>6} "
              f"{r['rejected']:>8} {'OK' if r['synced'] else 'DESYNC':>6}  {r['outcome']}")
        if not r["synced"]:
            print(f"    agent: {r['agent_fen']}")
            print(f"    site : {r['site_fen']}")
    print()
    print("All games stayed in sync." if not failures
          else f"{failures}/{games} games had problems.")
    return 0 if not failures else 1
