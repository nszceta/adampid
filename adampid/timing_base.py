from typing import Protocol, runtime_checkable

@runtime_checkable
class TimerBase(Protocol):
    def get_time_us(self) -> float:
        """Get current time in microseconds"""
        ...
    
    def get_time_ms(self) -> float:
        """Get current time in milliseconds"""
        ...
    
    def get_time_s(self) -> float:
        """Get current time in seconds"""
        ...