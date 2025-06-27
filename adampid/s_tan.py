"""
Sliding Tangent (STan) - Circular buffer for tangent line calculations.

This module implements a circular buffer that maintains a sliding window of 
values and efficiently calculates average values and tangent slopes. It's used
by the sTune autotuner for inflection point detection.
"""

from typing import List, Optional


class STan:
    """
    Circular buffer for sliding tangent line calculations.
    
    Maintains a fixed-size circular buffer of readings and provides efficient
    calculation of average values and slopes across the buffer window. This is
    essential for the inflection point detection algorithm in autotuning.
    
    The buffer operates as a FIFO (First In, First Out) queue where new values
    replace the oldest values when the buffer is full.
    """
    
    def __init__(self, buffer_size: Optional[int] = None):
        """
        Initialize the sliding tangent buffer.
        
        Args:
            buffer_size: Size of the circular buffer. If None, must call begin() later.
        """
        self._buffer_size: int = 0
        self._index: int = 0
        self._sum: float = 0.0
        self._input_array: List[float] = []
        
        if buffer_size is not None:
            self.begin(buffer_size)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup buffer."""
        self.clear()
    
    def begin(self, buffer_size: int) -> None:
        """
        Initialize the buffer with specified size.
        
        Args:
            buffer_size: Number of elements the buffer can hold
            
        Raises:
            ValueError: If buffer_size is not positive
        """
        if buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
            
        self._buffer_size = buffer_size
        self._input_array = [0.0] * buffer_size
        self.init(0.0)
    
    def init(self, reading: float) -> None:
        """
        Initialize all buffer elements with the same value.
        
        This is typically called with the first sensor reading to avoid
        artificial transients in the average calculation.
        
        Args:
            reading: Value to fill the entire buffer with
        """
        if not self._input_array:
            raise RuntimeError("Buffer not initialized. Call begin() first.")
            
        self._index = 0
        self._sum = reading * self._buffer_size
        for i in range(self._buffer_size):
            self._input_array[i] = reading
    
    def avg_val(self, reading: float) -> float:
        """
        Add a new reading and return the current buffer average.
        
        This implements an efficient rolling average using the circular buffer.
        The algorithm maintains a running sum and updates it by subtracting the
        value being replaced and adding the new value.
        
        Args:
            reading: New sensor reading to add to buffer
            
        Returns:
            Current average of all values in the buffer
        """
        if not self._input_array:
            raise RuntimeError("Buffer not initialized. Call begin() first.")
        
        # Move to next position (circular)
        self._index += 1
        if self._index >= self._buffer_size:
            self._index = 0
        
        # Update running sum efficiently: subtract old value, add new value
        self._sum += reading - self._input_array[self._index]
        self._input_array[self._index] = reading
        
        # Return current average
        return self._sum / self._buffer_size
    
    def start_val(self) -> float:
        """
        Get the oldest value in the buffer (tail of the sliding window).
        
        This represents the starting point of the current tangent line across
        the buffer window.
        
        Returns:
            The oldest value currently in the buffer
        """
        if not self._input_array:
            raise RuntimeError("Buffer not initialized. Call begin() first.")
        
        # The tail is the next position that will be overwritten
        tail_index = self._index + 1
        if tail_index >= self._buffer_size:
            tail_index = 0
            
        return self._input_array[tail_index]
    
    def slope(self, reading: float) -> float:
        """
        Calculate the slope from the oldest value to the current reading.
        
        This gives the slope of a line drawn from the start of the buffer
        window to the current point, representing the tangent across the
        entire buffer span.
        
        Args:
            reading: Current reading (most recent value)
            
        Returns:
            Slope from buffer start to current reading
        """
        return reading - self.start_val()
    
    def length(self) -> int:
        """
        Get the buffer size.
        
        Returns:
            Number of elements the buffer can hold
        """
        return self._buffer_size
    
    def clear(self) -> None:
        """Clear the buffer and reset all values."""
        self._index = 0
        self._sum = 0.0
        if self._input_array:
            for i in range(len(self._input_array)):
                self._input_array[i] = 0.0
    
    def is_initialized(self) -> bool:
        """Check if the buffer has been properly initialized."""
        return len(self._input_array) > 0
    
    def get_current_average(self) -> float:
        """
        Get the current buffer average without adding a new reading.
        
        Returns:
            Current average of all values in buffer
        """
        if not self._input_array:
            raise RuntimeError("Buffer not initialized. Call begin() first.")
            
        return self._sum / self._buffer_size
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self._input_array:
            return (f"STan(size={self._buffer_size}, index={self._index}, "
                   f"avg={self.get_current_average():.3f})")
        else:
            return "STan(uninitialized)"