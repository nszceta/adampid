"""
Manual step timer implementation for simulation
"""

from .timing_base import TimerBase


class SimulatedTimer(TimerBase):
    def __init__(self):
        self._current_time_us = 0.0

    def step(self, delta_us: float):
        """Manually advance time by delta in microseconds"""
        self._current_time_us += delta_us

    def reset(self):
        """Reset timer to zero"""
        self._current_time_us = 0.0

    def get_time_us(self) -> float:
        return self._current_time_us

    def get_time_ms(self) -> float:
        return self._current_time_us / 1000.0

    def get_time_s(self) -> float:
        return self._current_time_us / 1_000_000.0

    def __str__(self):
        return f"SimulatedTimer({self._current_time_us}us)"
