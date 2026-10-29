"""Stockfish wrapper.

The agent's chess strength lives here and nowhere else. An LLM is the wrong
tool for picking moves -- it plays around 1500 and hallucinates illegal ones;
a 100MB binary plays at 3600 and answers in milliseconds.
"""
from __future__ import annotations

import random

import chess
import chess.engine

from .config import ENGINE_PATH


class Brain:
    """A configured Stockfish process.

    `strength` is 0-20 (Stockfish's own Skill Level scale) or None for full
    power. Weakening it is the difference between a useful sparring partner
    and something that wins every game by move 25.
    """

    def __init__(
        self,
        strength: int | None = None,
        think_time: float = 0.5,
        depth: int | None = None,
        engine_path: str = ENGINE_PATH,
        threads: int = 2,
        hash_mb: int = 256,
    ):
        self.think_time = think_time
        self.depth = depth
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

        options = {}
        available = self.engine.options
        if "Threads" in available:
            options["Threads"] = threads
        if "Hash" in available:
            options["Hash"] = hash_mb
        if strength is not None and "Skill Level" in available:
            options["Skill Level"] = max(0, min(20, strength))
        if options:
            self.engine.configure(options)

        self.name = self.engine.id.get("name", "engine")
        self.strength = strength

    @property
    def _limit(self) -> chess.engine.Limit:
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        # Vary the clock a little so move timing does not look metronomic.
        t = self.think_time * random.uniform(0.75, 1.35)
        return chess.engine.Limit(time=t)

    def best_move(self, board: chess.Board) -> tuple[chess.Move, str]:
        """-> (move, short evaluation string)."""
        info = self.engine.analyse(board, self._limit)
        pv = info.get("pv") or []
        move = pv[0] if pv else self.engine.play(board, self._limit).move

        score = info.get("score")
        if score is None:
            note = "?"
        else:
            pov = score.white() if board.turn == chess.WHITE else score.black()
            mate = pov.mate()
            note = f"mate in {abs(mate)}" if mate is not None else f"{pov.score(mate_score=100000) / 100:+.2f}"
        return move, note

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
