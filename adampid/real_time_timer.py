"""
Wall-clock time implementation using system clocks
"""
import time
from .timing_base import TimerBase

class RealTimeTimer(TimerBase):
    def __init__(self):
        self._timer_func = self._select_timer()
        self._timer_name = self._timer_func.__name__
        self._is_nanosecond = 'ns' in self._timer_name
    
    def _select_timer(self):
        if hasattr(time, 'monotonic_ns'):
            try:
                time.monotonic_ns()
                return time.monotonic_ns
            except Exception:
                pass
        if hasattr(time, 'perf_counter_ns'):
            try:
                time.perf_counter_ns()
                return time.perf_counter_ns
            except Exception:
                pass
        if hasattr(time, 'time_ns'):
            try:
                time.time_ns()
                return time.time_ns
            except Exception:
                pass
        return time.time

    def get_time_us(self) -> float:
        if self._is_nanosecond:
            return self._timer_func() / 1000.0
        return self._timer_func() * 1_000_000.0
    
    def get_time_ms(self) -> float:
        if self._is_nanosecond:
            return self._timer_func() / 1_000_000.0
        return self._timer_func() * 1000.0
    
    def get_time_s(self) -> float:
        if self._is_nanosecond:
            return self._timer_func() / 1_000_000_000.0
        return self._timer_func()
    
    def __str__(self):
        return f"RealTimeTimer({self._timer_name})"