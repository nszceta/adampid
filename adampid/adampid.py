"""
AdamPID - Advanced PID Controller

This module provides a comprehensive PID controller implementation with advanced
features including multiple calculation modes, anti-windup protection, and
flexible configuration options. It maintains compatibility with the Arduino
PID library while adding enhanced functionality.

The controller supports various proportional and derivative calculation methods,
multiple anti-windup strategies, and automatic/manual/timer operation modes.
"""

from enum import IntEnum
from typing import Optional, Callable

from .timing_base import TimerBase
from .real_time_timer import RealTimeTimer
from .exceptions import ConfigurationError


class Control(IntEnum):
    """PID controller operation modes."""

    MANUAL = 0  # Manual mode - controller output not updated
    AUTOMATIC = 1  # Automatic mode - PID calculation active
    TIMER = 2  # Timer mode - calculate on timer intervals only
    TOGGLE = 3  # Toggle between manual and automatic


class Action(IntEnum):
    """Controller action types."""

    DIRECT = 0  # Direct acting - positive error increases output
    REVERSE = 1  # Reverse acting - positive error decreases output


class PMode(IntEnum):
    """Proportional term calculation modes."""

    P_ON_ERROR = 0  # Proportional on error (traditional)
    P_ON_MEAS = 1  # Proportional on measurement (reduces overshoot)
    P_ON_ERROR_MEAS = 2  # Average of error and measurement methods


class DMode(IntEnum):
    """Derivative term calculation modes."""

    D_ON_ERROR = 0  # Derivative on error
    D_ON_MEAS = 1  # Derivative on measurement (reduces derivative kick)


class IAwMode(IntEnum):
    """Integral anti-windup modes."""

    I_AW_CONDITION = 0  # Conditional integration (default)
    I_AW_CLAMP = 1  # Clamp output after integration
    I_AW_OFF = 2  # No anti-windup protection


class AdamPID:
    """
    Advanced PID Controller with multiple calculation modes and anti-windup.

    This controller provides sophisticated PID control with configurable
    calculation methods for proportional and derivative terms, multiple
    anti-windup strategies, and flexible operation modes.

    Key features:
    - Multiple proportional calculation modes (on error, measurement, or both)
    - Derivative calculation on error or measurement to reduce derivative kick
    - Advanced anti-windup protection with multiple strategies
    - Automatic, manual, and timer-based operation modes
    - Bumpless transfer between manual and automatic modes
    - Configurable output limits and sample timing
    """

    def __init__(
        self,
        input_var: Optional[Callable[[], float]] = None,
        output_var: Optional[Callable[[float], None]] = None,
        setpoint_var: Optional[Callable[[], float]] = None,
        kp: float = 0.0,
        ki: float = 0.0,
        kd: float = 0.0,
        p_mode: PMode = PMode.P_ON_ERROR,
        d_mode: DMode = DMode.D_ON_MEAS,
        i_aw_mode: IAwMode = IAwMode.I_AW_CONDITION,
        action: Action = Action.DIRECT,
        timer: Optional[TimerBase] = None,
    ):
        """
        Initialize the AdamPID controller.

        Args:
            input_var: Function that returns current process variable
            output_var: Function that sets the control output
            setpoint_var: Function that returns current setpoint
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            p_mode: Proportional calculation mode
            d_mode: Derivative calculation mode
            i_aw_mode: Integral anti-windup mode
            action: Controller action (direct or reverse)
        """
        # External variable connections
        self._input_var = input_var
        self._output_var = output_var
        self._setpoint_var = setpoint_var

        # Controller mode and configuration
        self._mode = Control.MANUAL
        self._action = action
        self._p_mode = p_mode
        self._d_mode = d_mode
        self._i_aw_mode = i_aw_mode

        # Tuning parameters (for display/query)
        self._disp_kp = kp
        self._disp_ki = ki
        self._disp_kd = kd

        # Internal working parameters (scaled by sample time)
        self._kp: float = 0.0
        self._ki: float = 0.0
        self._kd: float = 0.0

        # PID term components
        self._p_term: float = 0.0
        self._i_term: float = 0.0
        self._d_term: float = 0.0
        self.output_sum: float = 0.0  # Public for external access

        # State variables
        self._error: float = 0.0
        self._last_error: float = 0.0
        self._last_input: float = 0.0

        # Store timer implementation
        self.timer = timer or RealTimeTimer()
        
        # Initialize timing with dependency-injected timer
        self._sample_time_us: int = 100_000
        self._last_time: float = self.timer.get_time_us() - self._sample_time_us

        # Output limits
        self._out_min: float = 0.0
        self._out_max: float = 255.0  # Arduino PWM default

        # Initialize timing and internal parameters
        self.set_output_limits(0.0, 255.0)
        self.set_tunings(kp, ki, kd, p_mode, d_mode, i_aw_mode)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.reset()

    def compute(self) -> bool:
        """
        Perform PID calculation if it's time for a new computation.

        This is the main PID calculation method that should be called regularly
        (typically every loop iteration). It automatically handles timing and
        only performs calculations when necessary based on the sample time.

        Returns:
            True if calculation was performed, False if not time yet

        Raises:
            RuntimeError: If input/output variables are not set
        """
        if self._mode == Control.MANUAL:
            return False

        if not all([self._input_var, self._output_var, self._setpoint_var]):
            raise RuntimeError("Input, output, and setpoint variables must be set")

        now = self.timer.get_time_us()
        time_change = now - self._last_time

        # Check if it's time to compute (timer mode or enough time elapsed)
        if self._mode == Control.TIMER or time_change >= self._sample_time_us:
            return self._perform_calculation(now)

        return False

    def _perform_calculation(self, now: float) -> bool:
        """Perform the actual PID calculation."""
        # Validate initialization
        assert self._input_var is not None
        assert self._setpoint_var is not None
        assert self._output_var is not None

        # Get current values
        input_val = self._input_var()
        setpoint = self._setpoint_var()

        # Calculate input change (for derivative and proportional on measurement)
        d_input = input_val - self._last_input
        if self._action == Action.REVERSE:
            d_input = -d_input

        # Calculate error
        self._error = setpoint - input_val
        if self._action == Action.REVERSE:
            self._error = -self._error

        d_error = self._error - self._last_error

        # Calculate proportional terms
        p_error_term = self._kp * self._error
        p_meas_term = self._kp * d_input

        # Apply proportional mode
        if self._p_mode == PMode.P_ON_ERROR:
            p_meas_term = 0
        elif self._p_mode == PMode.P_ON_MEAS:
            p_error_term = 0
        else:  # P_ON_ERROR_MEAS - average both
            p_error_term *= 0.5
            p_meas_term *= 0.5

        # Store proportional term for debugging
        self._p_term = p_error_term - p_meas_term

        # Calculate integral term
        self._i_term = self._ki * self._error

        # Calculate derivative term based on mode
        if self._d_mode == DMode.D_ON_ERROR:
            self._d_term = self._kd * d_error
        else:  # D_ON_MEAS
            self._d_term = -self._kd * d_input

        # Apply anti-windup for integral term
        if self._i_aw_mode == IAwMode.I_AW_CONDITION:
            self._apply_conditional_anti_windup(p_error_term, p_meas_term, d_error)

        # Update output sum with integral term
        self.output_sum += self._i_term

        # Apply output limits to sum (except for no anti-windup mode)
        if self._i_aw_mode == IAwMode.I_AW_OFF:
            self.output_sum -= p_meas_term  # Include p_meas_term without limits
        else:
            self.output_sum = self._constrain(
                self.output_sum - p_meas_term, self._out_min, self._out_max
            )

        # Calculate final output
        output = self.output_sum + p_error_term + self._d_term
        output = self._constrain(output, self._out_min, self._out_max)

        # Set output
        self._output_var(output)

        # Update state for next iteration
        self._last_error = self._error
        self._last_input = input_val
        self._last_time = now

        return True

    def _apply_conditional_anti_windup(
        self, p_error_term: float, p_meas_term: float, d_error: float
    ) -> None:
        """
        Apply conditional anti-windup protection.

        This method prevents integral windup by checking if the output would
        saturate and the error is driving in the same direction. If so, it
        clamps the integral term to prevent further windup.
        """
        aw = False
        i_term_out = (p_error_term - p_meas_term) + self._ki * (
            self._i_term + self._error
        )

        if i_term_out > self._out_max and d_error > 0:
            aw = True
        elif i_term_out < self._out_min and d_error < 0:
            aw = True

        if aw and self._ki != 0:
            self._i_term = self._constrain(i_term_out, -self._out_max, self._out_max)

    def set_tunings(
        self,
        kp: float,
        ki: float,
        kd: float,
        p_mode: Optional[PMode] = None,
        d_mode: Optional[DMode] = None,
        i_aw_mode: Optional[IAwMode] = None,
    ) -> None:
        """
        Set PID tuning parameters and calculation modes.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            p_mode: Proportional calculation mode (optional)
            d_mode: Derivative calculation mode (optional)
            i_aw_mode: Integral anti-windup mode (optional)

        Raises:
            ConfigurationError: If any gain is negative
        """
        if kp < 0 or ki < 0 or kd < 0:
            raise ConfigurationError("PID gains cannot be negative")

        # Reset integral sum if Ki changes to zero
        if ki == 0:
            self.output_sum = 0

        # Update modes if provided
        if p_mode is not None:
            self._p_mode = p_mode
        if d_mode is not None:
            self._d_mode = d_mode
        if i_aw_mode is not None:
            self._i_aw_mode = i_aw_mode

        # Store display values
        self._disp_kp = kp
        self._disp_ki = ki
        self._disp_kd = kd

        # Calculate internal working parameters (scaled by sample time)
        sample_time_sec = self._sample_time_us / 1_000_000.0
        self._kp = kp
        self._ki = ki * sample_time_sec
        self._kd = kd / sample_time_sec

    def set_sample_time_us(self, new_sample_time_us: int) -> None:
        """
        Set the sample time in microseconds.

        The sample time determines how often the PID calculation is performed.
        Changing the sample time automatically rescales the integral and
        derivative terms to maintain the same control behavior.

        Args:
            new_sample_time_us: New sample time in microseconds

        Raises:
            ConfigurationError: If sample time is not positive
        """
        if new_sample_time_us <= 0:
            raise ConfigurationError("Sample time must be positive")

        # Calculate scaling ratio
        ratio = new_sample_time_us / self._sample_time_us

        # Rescale integral and derivative terms
        self._ki *= ratio
        self._kd /= ratio

        self._sample_time_us = new_sample_time_us

    def set_output_limits(self, min_output: float, max_output: float) -> None:
        """
        Set the output limits for the PID controller.

        Args:
            min_output: Minimum output value
            max_output: Maximum output value

        Raises:
            ConfigurationError: If min >= max
        """
        if min_output >= max_output:
            raise ConfigurationError("Minimum output must be less than maximum")

        self._out_min = min_output
        self._out_max = max_output

        # Constrain current output and sum if in automatic mode
        if self._mode != Control.MANUAL and self._output_var:
            current_output = self._get_current_output()
            if current_output is not None:
                constrained_output = self._constrain(
                    current_output, min_output, max_output
                )
                self._output_var(constrained_output)

            self.output_sum = self._constrain(self.output_sum, min_output, max_output)

    def set_mode(self, mode: Control) -> None:
        """
        Set the controller operation mode.

        Args:
            mode: New operation mode
        """
        # Handle transition from manual to automatic/timer
        if self._mode == Control.MANUAL and mode != Control.MANUAL:
            self.initialize()

        # Handle toggle mode
        if mode == Control.TOGGLE:
            self._mode = (
                Control.AUTOMATIC if self._mode == Control.MANUAL else Control.MANUAL
            )
        else:
            self._mode = mode

    def set_controller_direction(self, action: Action) -> None:
        """Set controller action (direct or reverse)."""
        self._action = action

    def set_proportional_mode(self, p_mode: PMode) -> None:
        """Set proportional calculation mode."""
        self._p_mode = p_mode

    def set_derivative_mode(self, d_mode: DMode) -> None:
        """Set derivative calculation mode."""
        self._d_mode = d_mode

    def set_anti_windup_mode(self, i_aw_mode: IAwMode) -> None:
        """Set integral anti-windup mode."""
        self._i_aw_mode = i_aw_mode

    def set_output_sum(self, sum_value: float) -> None:
        """Set the integral sum value directly."""
        self.output_sum = sum_value

    def set_input_output_setpoint(
        self,
        input_var: Callable[[], float],
        output_var: Callable[[float], None],
        setpoint_var: Callable[[], float],
    ) -> None:
        """Set input, output, and setpoint variable functions."""
        self._input_var = input_var
        self._output_var = output_var
        self._setpoint_var = setpoint_var

    def initialize(self) -> None:
        """
        Initialize the controller for bumpless transfer from manual to automatic.

        This method should be called when switching from manual to automatic mode
        to prevent output bumps. It initializes the integral sum and last input
        to current values.
        """
        if self._input_var:
            self._last_input = self._input_var()

        current_output = self._get_current_output()
        if current_output is not None:
            self.output_sum = self._constrain(
                current_output, self._out_min, self._out_max
            )
        else:
            self.output_sum = 0.0

    def reset(self) -> None:
        """Reset all internal state variables to zero."""
        self._last_time = self.timer.get_time_us() - self._sample_time_us
        self._last_input = 0.0
        self.output_sum = 0.0
        self._p_term = 0.0
        self._i_term = 0.0
        self._d_term = 0.0
        self._error = 0.0
        self._last_error = 0.0

    def _get_current_output(self) -> Optional[float]:
        """Get current output value if available."""
        # Since we can't read back from a function, return None
        # In practice, the user would need to maintain this separately
        return None

    def _constrain(self, value: float, min_val: float, max_val: float) -> float:
        """Constrain a value between min and max."""
        return max(min_val, min(max_val, value))

    # Query methods
    def get_kp(self) -> float:
        """Get proportional gain."""
        return self._disp_kp

    def get_ki(self) -> float:
        """Get integral gain."""
        return self._disp_ki

    def get_kd(self) -> float:
        """Get derivative gain."""
        return self._disp_kd

    def get_p_term(self) -> float:
        """Get proportional term component."""
        return self._p_term

    def get_i_term(self) -> float:
        """Get integral term component."""
        return self._i_term

    def get_d_term(self) -> float:
        """Get derivative term component."""
        return self._d_term

    def get_output_sum(self) -> float:
        """Get integral sum value."""
        return self.output_sum

    def get_mode(self) -> int:
        """Get current operation mode as integer."""
        return int(self._mode)

    def get_direction(self) -> int:
        """Get controller action as integer."""
        return int(self._action)

    def get_p_mode(self) -> int:
        """Get proportional mode as integer."""
        return int(self._p_mode)

    def get_d_mode(self) -> int:
        """Get derivative mode as integer."""
        return int(self._d_mode)

    def get_aw_mode(self) -> int:
        """Get anti-windup mode as integer."""
        return int(self._i_aw_mode)

    def get_sample_time_us(self) -> int:
        """Get sample time in microseconds."""
        return self._sample_time_us

    def get_output_limits(self) -> tuple[float, float]:
        """Get output limits as (min, max) tuple."""
        return (self._out_min, self._out_max)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AdamPID(Kp={self._disp_kp:.3f}, Ki={self._disp_ki:.3f}, "
            f"Kd={self._disp_kd:.3f}, mode={self._mode.name})"
        )
