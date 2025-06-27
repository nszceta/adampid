"""
High-precision timing utilities for AdamPID.

Implements a fallback hierarchy of timing functions, starting with the most
precise available and falling back to less precise alternatives. The timing
function is selected once at module import and cached for performance.
"""

import time
from typing import Callable


class Timer:
    """
    High-precision timer with automatic fallback to best available timing method.
    
    Attempts to use timing functions in order of preference:
    1. time.monotonic_ns() - Monotonic nanosecond precision (Python 3.7+)
    2. time.perf_counter_ns() - Performance counter nanosecond precision (Python 3.7+) 
    3. time.time_ns() - System time nanosecond precision (Python 3.7+)
    4. time.time() - System time float seconds (fallback)
    
    The selected timing function is cached for performance.
    """
    
    def __init__(self):
        self._timer_func: Callable[[], float] = self._select_timer()
        self._timer_name: str = self._timer_func.__name__
        self._is_nanosecond: bool = 'ns' in self._timer_name
        
    def _select_timer(self) -> Callable[[], float]:
        """
        Select the best available timing function through systematic testing.
        
        Returns:
            Callable that returns time in either nanoseconds or seconds
        """
        # Try monotonic_ns first - best for relative timing, immune to system clock changes
        if hasattr(time, 'monotonic_ns'):
            try:
                time.monotonic_ns()  # Test call
                return time.monotonic_ns
            except (AttributeError, OSError):
                pass
        
        # Try perf_counter_ns - high resolution performance counter
        if hasattr(time, 'perf_counter_ns'):
            try:
                time.perf_counter_ns()  # Test call
                return time.perf_counter_ns
            except (AttributeError, OSError):
                pass
        
        # Try time_ns - nanosecond system time
        if hasattr(time, 'time_ns'):
            try:
                time.time_ns()  # Test call
                return time.time_ns
            except (AttributeError, OSError):
                pass
        
        # Final fallback - standard time.time()
        return time.time
    
    def get_time_us(self) -> float:
        """
        Get current time in microseconds as float.
        
        Returns:
            Current time in microseconds
        """
        if self._is_nanosecond:
            return self._timer_func() / 1000.0  # Convert ns to μs
        else:
            return self._timer_func() * 1_000_000.0  # Convert s to μs
    
    def get_time_ms(self) -> float:
        """
        Get current time in milliseconds as float.
        
        Returns:
            Current time in milliseconds
        """
        if self._is_nanosecond:
            return self._timer_func() / 1_000_000.0  # Convert ns to ms
        else:
            return self._timer_func() * 1000.0  # Convert s to ms
    
    def get_time_s(self) -> float:
        """
        Get current time in seconds as float.
        
        Returns:
            Current time in seconds
        """
        if self._is_nanosecond:
            return self._timer_func() / 1_000_000_000.0  # Convert ns to s
        else:
            return self._timer_func()
    
    @property
    def timer_name(self) -> str:
        """Name of the selected timing function."""
        return self._timer_name
    
    @property
    def resolution_description(self) -> str:
        """Human-readable description of timing resolution."""
        if self._timer_name == 'monotonic_ns':
            return "Monotonic nanosecond precision"
        elif self._timer_name == 'perf_counter_ns':
            return "Performance counter nanosecond precision"
        elif self._timer_name == 'time_ns':
            return "System time nanosecond precision"
        else:
            return "System time float seconds (fallback)"


# Global timer instance - initialized once at module import
_timer = Timer()

# Convenience functions that use the global timer
def get_time_us() -> float:
    """Get current time in microseconds."""
    return _timer.get_time_us()

def get_time_ms() -> float:
    """Get current time in milliseconds.""" 
    return _timer.get_time_ms()

def get_time_s() -> float:
    """Get current time in seconds."""
    return _timer.get_time_s()

def get_timer_info() -> dict:
    """Get information about the selected timer."""
    return {
        'function': _timer.timer_name,
        'description': _timer.resolution_description,
        'is_nanosecond': _timer._is_nanosecond
    }