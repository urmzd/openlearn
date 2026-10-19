"""A fake clock for contracts that take a Clock (no real sleeps
in tests). Code under test calls clock.now(), clock.monotonic(), and
clock.sleep(s); a FakeClock moves only when told, and its sleep advances
time at once while recording what was asked."""

from __future__ import annotations

import time


class RealClock:
    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock:
    def __init__(self, start: float = 1_767_225_600.0):  # 2026-01-01T00:00:00Z
        self._wall = float(start)
        self._mono = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self._wall

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a clock never goes backwards (monotonic time)")
        self._wall += seconds
        self._mono += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(max(0.0, seconds))
