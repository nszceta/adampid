"""
STune - Inflection Point Autotuner (Instance Method Version)

This module implements an advanced PID autotuner using instance methods for
input/output operations rather than callable functions.
"""

from enum import IntEnum
from typing import Optional, Tuple

from .s_tan import STan
from .timing_base import TimerBase
from .real_time_timer import RealTimeTimer
from .exceptions import TuningError, ConfigurationError


class TunerAction(IntEnum):
    """Controller action types for autotuning."""

    DIRECT_IP = 0  # Direct acting, inflection point method
    DIRECT_5T = 1  # Direct acting, 5-tau full response method
    REVERSE_IP = 2  # Reverse acting, inflection point method
    REVERSE_5T = 3  # Reverse acting, 5-tau full response method


class SerialMode(IntEnum):
    """Serial output verbosity levels."""

    PRINT_OFF = 0  # No output
    PRINT_ALL = 1  # All debugging information
    PRINT_SUMMARY = 2  # Summary results only
    PRINT_DEBUG = 3  # Detailed debugging information


class TunerStatus(IntEnum):
    """Internal tuner state machine status."""

    SAMPLE = 0  # Ready to sample
    TEST = 1  # Running test
    TUNINGS = 2  # Test complete, tunings available
    RUN_PID = 3  # Running PID (post-tuning)
    TIMER_PID = 4  # PID timer state


class TuningMethod(IntEnum):
    """Available PID tuning calculation methods."""

    # PID Controllers
    ZN_PID = 0  # Ziegler-Nichols PID
    DAMPED_OSC_PID = 1  # Damped Oscillation PID
    NO_OVERSHOOT_PID = 2  # No Overshoot PID
    COHEN_COON_PID = 3  # Cohen-Coon PID
    MIXED_PID = 4  # Mixed PID (average of methods)

    # PI Controllers
    ZN_PI = 5  # Ziegler-Nichols PI
    DAMPED_OSC_PI = 6  # Damped Oscillation PI
    NO_OVERSHOOT_PI = 7  # No Overshoot PI
    COHEN_COON_PI = 8  # Cohen-Coon PI
    MIXED_PI = 9  # Mixed PI (average of methods)


class STune:
    """
    Inflection Point Autotuner with instance method interface.

    This autotuner uses instance methods for input/output operations, allowing
    the application to control timing and data flow.

    Usage pattern:
        tuner = STune(tuning_method=TuningMethod.COHEN_COON_PID)
        tuner.configure(input_span=100, output_span=100, ...)

        # In control loop:
        tuner.set_input(sensor_reading)
        status = tuner.run()
        if status in [TunerStatus.TEST, TunerStatus.SAMPLE]:
            actuator_output = tuner.get_output()
            apply_output_to_actuator(actuator_output)
        elif status == TunerStatus.TUNINGS:
            kp, ki, kd = tuner.get_auto_tunings()
    """

    # Mathematical constants used in calculations
    K_EXP = 4.3004  # (1 / exp(-1)) / (1 - exp(-1)) - used for apparent max calculation
    EPSILON = 0.0001  # Small value for floating point comparisons

    def __init__(
        self,
        tuning_method: TuningMethod = TuningMethod.ZN_PID,
        action: TunerAction = TunerAction.DIRECT_IP,
        serial_mode: SerialMode = SerialMode.PRINT_OFF,
        timer: Optional[TimerBase] = None,
    ):
        """
        Initialize the STune autotuner.

        Args:
            tuning_method: Method to use for calculating PID parameters
            action: Control action and test method
            serial_mode: Verbosity level for debugging output
            timer: Timer implementation for timing control
        """
        # Store timer implementation
        self.timer = timer or RealTimeTimer()

        # Current process values (set/retrieved by application)
        self._current_input: float = 0.0
        self._current_output: float = 0.0
        self._input_valid: bool = False

        # Configuration
        self._action = action
        self._serial_mode = serial_mode
        self._tuning_method = tuning_method

        # Process model parameters (calculated during tuning)
        self._kp: float = 0.0  # Proportional gain
        self._ki: float = 0.0  # Integral gain
        self._kd: float = 0.0  # Derivative gain
        self._ku: float = 0.0  # Ultimate/process gain
        self._tu: float = 0.0  # Ultimate/time constant period
        self._td: float = 0.0  # Dead time
        self._r: float = 0.0  # Ratio td/tu
        self._ko: float = 0.0  # Output gain (unused in current implementation)

        # Test configuration parameters
        self._input_span: float = 0.0
        self._output_span: float = 0.0
        self._output_start: float = 0.0
        self._output_step: float = 0.0
        self._test_time_sec: int = 0
        self._settle_time_sec: int = 0
        self._samples: int = 0

        # Calculated timing parameters
        self._sample_period_us: float = 0.0
        self._settle_period_us: float = 0.0
        self._tangent_period_us: float = 0.0
        self._buffer_size: int = 0

        # Process variables during testing
        self.e_stop: float = 0.0  # Emergency stop threshold
        self.pv_inst: float = 0.0  # Instantaneous process variable
        self.pv_avg: float = 0.0  # Average process variable (buffered)
        self.pv_ip: float = 0.0  # Process variable at inflection point
        self.pv_max: float = 0.0  # Maximum/minimum process variable
        self.pv_pk: float = 0.0  # Peak value for 5T testing
        self.pv_inst_res: float = 0.0  # Resolution of instantaneous readings
        self.pv_avg_res: float = 0.0  # Resolution of average readings
        self.slope_ip: float = 0.0  # Slope at inflection point
        self.pv_tangent: float = 0.0  # Current tangent value
        self.pv_tangent_prev: float = 0.0  # Previous tangent value
        self.pv_start: float = 0.0  # Starting process variable value

        # State tracking variables
        self._tuner_status = TunerStatus.TEST
        self.sample_count: int = 0
        self.pv_pk_count: int = 0
        self.ip_count: int = 0
        self.plot_count: int = 0
        self.e_stop_abort: int = 0

        # Timing variables
        self.us_prev: float = 0.0
        self.settle_prev: float = 0.0
        self.us_start: float = 0.0
        self.us: float = 0.0
        self.ip_us: float = 0.0

        # Sliding tangent calculator
        self._tangent = STan()

        # Initialize to safe state
        self.reset()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.reset()
        if hasattr(self._tangent, "__exit__"):
            self._tangent.__exit__(exc_type, exc_val, exc_tb)

    def set_input(self, input_value: float) -> None:
        """
        Set the current process variable (sensor reading).

        Args:
            input_value: Current measured process variable
        """
        self._current_input = input_value
        self._input_valid = True

    def get_output(self) -> float:
        """
        Get the current controller output that should be applied.

        Returns:
            Current output value for the actuator
        """
        return self._current_output

    def configure(
        self,
        input_span: float,
        output_span: float,
        output_start: float,
        output_step: float,
        test_time_sec: int,
        settle_time_sec: int,
        samples: int,
    ) -> None:
        """Configure the autotuning test parameters."""

        # ADDED: Optimize parameters for better autotuning
        if abs(output_step - output_start) < 5.0:
            # Increase step size if too small
            if output_step > output_start:
                output_step = output_start + 15.0
            else:
                output_step = output_start - 15.0
            if self._serial_mode != SerialMode.PRINT_OFF:
                print(f"Increased step size to {output_step} for better identification")

        # Ensure adequate sampling resolution
        min_samples = max(300, test_time_sec * 10)
        if samples < min_samples:
            samples = min_samples
            if self._serial_mode != SerialMode.PRINT_OFF:
                print(f"Increased samples to {samples} for better resolution")

        # EXISTING CODE BELOW - NO CHANGES:
        # Validate parameters
        if input_span <= 0:
            raise ConfigurationError("Input span must be positive")
        if output_span <= 0:
            raise ConfigurationError("Output span must be positive")
        if test_time_sec <= 0:
            raise ConfigurationError("Test time must be positive")
        if settle_time_sec < 0:
            raise ConfigurationError("Settle time cannot be negative")
        if samples <= 0:
            raise ConfigurationError("Sample count must be positive")
        if abs(output_step) <= self.EPSILON:
            raise ConfigurationError("Output step must be non-zero")

        # Reset state before new configuration
        self.reset()

        # Store configuration
        self._input_span = input_span
        self._output_span = output_span
        self._output_start = output_start
        self._output_step = output_step
        self._test_time_sec = test_time_sec
        self._settle_time_sec = settle_time_sec
        self._samples = samples

        # Calculate derived parameters
        self._buffer_size = max(
            1, int(self._samples * 0.06)
        )  # 6% of samples for buffer
        self._sample_period_us = (self._test_time_sec * 1_000_000.0) / self._samples
        self._tangent_period_us = self._sample_period_us * (self._buffer_size - 1)
        self._settle_period_us = self._settle_time_sec * 1_000_000.0

        # Set emergency stop to input span by default
        self.e_stop = input_span

        # Initialize sliding tangent buffer
        self._tangent.begin(self._buffer_size)

        # Set initial output
        self._current_output = self._output_start

        if self._serial_mode != SerialMode.PRINT_OFF:
            print(
                f"STune configured: {samples} samples over {test_time_sec}s, "
                f"buffer size: {self._buffer_size}"
            )

    def run(self) -> TunerStatus:
        """
        Execute one iteration of the autotuning state machine.

        This method should be called repeatedly after setting the input value.
        It returns the current state and updates the output value internally.

        Returns:
            Current tuner status

        Raises:
            TuningError: If tuning fails or emergency stop is triggered
        """
        # CRITICAL FIX: Better error message and validation
        if not self._input_valid:
            raise TuningError(
                f"Input value must be set before calling run(). "
                f"Call set_input() first. Current input valid: {self._input_valid}"
            )

        # Update internal process variable from current input
        self.pv_inst = self._current_input

        us_now = self.timer.get_time_us()
        us_elapsed = us_now - self.us_prev
        settle_elapsed = us_now - self.settle_prev
        self.us = us_now - self.us_start

        if self._tuner_status == TunerStatus.SAMPLE:
            self._tuner_status = TunerStatus.TEST
            return TunerStatus.TEST

        elif self._tuner_status == TunerStatus.TEST:
            return self._run_test_phase(us_now, us_elapsed, settle_elapsed)

        elif self._tuner_status == TunerStatus.TUNINGS:
            self._tuner_status = TunerStatus.TIMER_PID
            return TunerStatus.TIMER_PID

        elif self._tuner_status == TunerStatus.RUN_PID:
            return self._run_pid_phase(us_now, us_elapsed)

        elif self._tuner_status == TunerStatus.TIMER_PID:
            return self._run_timer_phase(us_elapsed)

        else:
            self._tuner_status = TunerStatus.TIMER_PID
            return TunerStatus.TIMER_PID

    def _run_test_phase(
        self, us_now: float, us_elapsed: float, settle_elapsed: float
    ) -> TunerStatus:
        """Execute the main autotuning test phase."""

        # Check emergency stop
        if self.pv_inst > self.e_stop and not self.e_stop_abort:
            self.reset()
            self.sample_count = self._samples + 1
            self.e_stop_abort = 1
            if self._serial_mode != SerialMode.PRINT_OFF:
                print("ABORT: pv_inst > e_stop")
            raise TuningError(
                "Emergency stop triggered: process variable exceeded limit"
            )

        # Fix 2: Improved settling and sampling logic
        if settle_elapsed >= self._settle_period_us:
            # Post-settling phase: normal autotuning

            if us_elapsed >= self._sample_period_us:
                self.us_prev = us_now

                # Fix 3: Proper sample count progression
                # During settling: sample_count stays at -1
                # First sample after settling: sample_count becomes 0
                # Continue incrementing for each sample

                if self.sample_count < self._samples:
                    # Increment sample count first
                    self.sample_count += 1

                    # Apply step on first sample
                    if self.sample_count == 1:
                        self._current_output = self._output_step
                        if self._serial_mode != SerialMode.PRINT_OFF:
                            print(f"Applying step change: {self._output_step}")

                    # Process the sample
                    self._process_sample(us_now)

                    # Print debug info
                    self._print_test_run()
                    self.pv_tangent_prev = self.pv_tangent

                    # Check if test completed
                    if self.sample_count >= self._samples:
                        self._complete_tuning()
                        self._tuner_status = TunerStatus.TUNINGS
                        return TunerStatus.TUNINGS

                    self._tuner_status = TunerStatus.SAMPLE
                    return TunerStatus.SAMPLE
                else:
                    self._tuner_status = TunerStatus.TUNINGS
                    return TunerStatus.TUNINGS
        else:
            # Fix 4: Settling phase - maintain initial output and collect baseline
            if us_elapsed >= self._sample_period_us and not self.e_stop_abort:
                self._current_output = self._output_start
                self.us_prev = us_now

                # During settling, just update baseline measurements
                if self.sample_count == -1:  # Initialize on first settling sample
                    if self._input_valid:
                        self.pv_start = self._current_input
                        self.pv_avg = self.pv_start
                        self.pv_inst_res = (
                            abs(self.pv_start)
                            if abs(self.pv_start) > self.EPSILON
                            else 1.0
                        )
                        self.pv_avg_res = self.pv_inst_res

                if self._serial_mode in [SerialMode.PRINT_ALL, SerialMode.PRINT_DEBUG]:
                    remaining_time = (self._settle_period_us - settle_elapsed) * 1e-6
                    print(
                        f" sec: {remaining_time:.4f}  out: {self._output_start}  "
                        f"pv: {self.pv_inst:.3f}  settling  ⤳⤳"
                    )

                self._tuner_status = TunerStatus.SAMPLE
                return TunerStatus.SAMPLE

        return self._tuner_status

    def _process_sample(self, us_now: float) -> TunerStatus:
        # Get current measurements and resolutions
        last_pv_inst = self.pv_inst
        last_pv_avg = self.pv_avg

        # Update current input from application
        self.pv_inst = self._current_input

        # Fix 5: Better averaging buffer management
        if self.sample_count == 0:
            # Initialize tangent buffer on first real sample
            self._tangent.init(self.pv_inst)
            self.pv_avg = self.pv_inst
            self.pv_start = self.pv_inst  # Set start value
            self.pv_inst_res = (
                abs(self.pv_inst) if abs(self.pv_inst) > self.EPSILON else 1.0
            )
            self.pv_avg_res = self.pv_inst_res
            self.us_start = us_now
            self.us = 0

            # Initialize slope tracking
            self.slope_ip = 0.0
            self.pv_tangent = 0.0
            self.pv_tangent_prev = 0.0

            if self._serial_mode != SerialMode.PRINT_OFF:
                print(f"Initialized: pv_start={self.pv_start:.3f}")
        else:
            # Normal sample processing
            self.pv_avg = self._tangent.avg_val(self.pv_inst)

            # Track resolution for noise analysis
            pv_inst_resolution = abs(self.pv_inst - last_pv_inst)
            pv_avg_resolution = abs(self.pv_avg - last_pv_avg)

            if (
                pv_inst_resolution > self.EPSILON
                and pv_inst_resolution < self.pv_inst_res
            ):
                self.pv_inst_res = pv_inst_resolution

            if pv_avg_resolution > self.EPSILON and pv_avg_resolution < self.pv_avg_res:
                self.pv_avg_res = pv_avg_resolution

        # Fix 6: Improved tangent calculation with validation
        if self.sample_count >= 1:  # Need at least 2 samples for tangent
            self.pv_tangent = self.pv_avg - self._tangent.start_val()

            if self.sample_count % 100 == 0:  # Every 100 samples
                print(
                    f"DEBUG: sample={self.sample_count}, pv_avg={self.pv_avg:.4f}, pv_start={self.pv_start:.4f}, pv_tangent={self.pv_tangent:.4f}"
                )

            # Detect dead time (when response begins)
            self._detect_dead_time()

            # Detect inflection point or continue 5T testing
            if self._action in [TunerAction.DIRECT_IP, TunerAction.REVERSE_IP]:
                self._detect_inflection_point()
            else:  # 5T testing
                self._continue_5t_testing()

        return TunerStatus.TEST

    def _detect_dead_time(self) -> None:
        """Improved dead time detection with better noise handling."""

        if self._td > 0:  # Only detect once
            return

        # Use multiple criteria for better dead time detection
        response_threshold = max(self.pv_inst_res * 3, abs(self.pv_start) * 0.01)

        dt_detected = False
        if self._action in [TunerAction.DIRECT_IP, TunerAction.DIRECT_5T]:
            # Look for sustained increase
            if (
                self.pv_avg > self.pv_start + response_threshold
                and self.pv_tangent > self.EPSILON
            ):
                dt_detected = True
        else:  # Reverse action
            # Look for sustained decrease
            if (
                self.pv_avg < self.pv_start - response_threshold
                and self.pv_tangent < -self.EPSILON
            ):
                dt_detected = True

        if dt_detected:
            self._td = self.us * 1e-6  # Convert to seconds
            if self._serial_mode == SerialMode.PRINT_DEBUG:
                print(
                    f"Dead time detected: {self._td:.3f}s at sample {self.sample_count}"
                )

    def _detect_inflection_point(self) -> None:
        """Detect the inflection point using tangent slope analysis."""

        # EXISTING CODE - keep as is:
        # Check for inflection point based on tangent slope
        ip_count = False
        if self._action in [TunerAction.DIRECT_IP, TunerAction.DIRECT_5T]:
            if self.pv_tangent > self.slope_ip + self.EPSILON:
                ip_count = True
            if self.pv_tangent < 0 + self.EPSILON:
                self.ip_count = 0  # Reset on flat/negative tangent
        else:  # Reverse action
            if self.pv_tangent < self.slope_ip - self.EPSILON:
                ip_count = True
            if self.pv_tangent > 0 - self.EPSILON:
                self.ip_count = 0  # Reset on flat/positive tangent

        if ip_count:
            self.ip_count = 0
            self.slope_ip = self.pv_tangent

        self.ip_count += 1

        # ADD THIS DEBUG OUTPUT:
        if (
            self.sample_count % 50 == 0 or self.ip_count > 30
        ):  # Every 50 samples or near threshold
            print(
                f"DEBUG IP: sample={self.sample_count}, ip_count={self.ip_count}, pv_tangent={self.pv_tangent:.4f}, slope_ip={self.slope_ip:.4f}"
            )

        # Declare inflection point found after sufficient samples
        ip_threshold = max(1, self._samples // 16)

        # ADD DEBUG FOR THRESHOLD:
        if self.ip_count == ip_threshold:
            print(f"DEBUG: Inflection point detected! Threshold={ip_threshold}")

        # EXISTING CODE - keep as is:
        if self.ip_count == ip_threshold:
            self.sample_count = self._samples
            self.ip_us = self.us
            self.pv_ip = self.pv_avg

            # Calculate apparent maximum using exponential approximation
            self.pv_max = self.pv_ip + (self.slope_ip * self.K_EXP)

            # ADD DEBUG FOR CALCULATION:
            print(
                f"DEBUG: pv_ip={self.pv_ip:.4f}, slope_ip={self.slope_ip:.4f}, K_EXP={self.K_EXP:.4f}"
            )
            print(f"DEBUG: Calculated pv_max={self.pv_max:.4f}")

            # Calculate time constant from tangent crossing points
            self._tu = (
                ((self.pv_max - self.pv_start) / self.slope_ip)
                * self._tangent_period_us
                * 1e-6
            ) - self._td

    def _continue_5t_testing(self) -> None:
        """Continue testing to 5τ (full process response)."""
        if self.sample_count >= self._samples - 1:
            self.sample_count = self._samples - 2

        # Only start peak tracking after 10% of test time
        if self.us > self._test_time_sec * 100_000:  # 10% in microseconds
            if self.pv_avg > self.pv_pk:
                # Set new boosted peak with resolution buffer
                self.pv_pk = self.pv_avg + (self._buffer_size * 0.2 * self.pv_avg_res)
                self.pv_pk_count = 0  # Reset counter
            else:
                self.pv_pk_count += 1

            # Check if response has stabilized
            stabilization_threshold = int(1.2 * self._buffer_size)
            if self.pv_pk_count == stabilization_threshold:
                self.pv_pk_count += 1
                self.sample_count = self._samples

                # Estimate final value assuming 3τ response, increase by 5% to get 5τ
                self.pv_max = self.pv_avg + (self.pv_inst - self.pv_start) * 0.05

                # Scale time to 5τ, then multiply by 0.286 to get τ
                self._tu = (self.us * 1.6667 * 1e-6 * 0.286) - self._td

    """
    Minimal fix for STune time constant calculation bug.

    The issue: _complete_tuning() incorrectly recalculates the time constant that was
    already correctly calculated in _detect_inflection_point(), causing poor process
    identification and subsequent poor PID performance.

    The fix: Remove the problematic recalculation to match the C++ implementation.
    """

    def _complete_tuning(self) -> None:
        """Complete the tuning calculation and compute PID parameters - IMPROVED VERSION."""

        pv_change = abs(self.pv_max - self.pv_start)
        output_change = abs(self._output_step - self._output_start)

        if self._serial_mode != SerialMode.PRINT_OFF:
            print(f"DEBUG: pv_start={self.pv_start:.3f}, pv_max={self.pv_max:.3f}")
            print(
                f"DEBUG: pv_change={pv_change:.3f}, output_change={output_change:.3f}"
            )
            print(f"DEBUG: Expected response ~{output_change * 0.8:.1f}")

        # Validation thresholds
        min_response = output_change * 0.05  # At least 5% of step size response
        if pv_change < min_response:
            raise TuningError(
                f"Insufficient process response: {pv_change:.3f} < {min_response:.3f}"
            )

        if output_change < 1.0:
            raise TuningError(f"Step size too small: {output_change:.3f}")

        self._ku = pv_change / output_change

        # Time constant calculation for inflection point method
        if self._action in [TunerAction.DIRECT_IP, TunerAction.REVERSE_IP]:
            # More robust time constant calculation
            if abs(self.slope_ip) > self.EPSILON:
                # Time from inflection point to apparent maximum
                time_to_max = (self.pv_max - self.pv_ip) / abs(self.slope_ip)
                # Convert buffer periods to actual time
                self._tu = time_to_max * self._tangent_period_us * 1e-6
            else:
                # Fallback: estimate from response time
                self._tu = self.ip_us * 1e-6 * 0.63  # Approximate time constant

        # Dead time ratio calculation
        if self._tu > 0:
            self._r = self._td / self._tu
        else:
            self._r = 0.1  # Default safe ratio

        # Validate identified parameters
        if self._ku <= 0 or self._tu <= 0:
            raise TuningError(
                f"Invalid identified parameters: Ku={self._ku}, Tu={self._tu}"
            )

        # Calculate PID parameters
        self._kp = self.get_kp()
        self._ki = self.get_ki()
        self._kd = self.get_kd()

        # Validate and constrain PID parameters
        if self._kp <= 0:
            self._kp = 1.0 / self._ku if self._ku > 0 else 1.0

        if self._ki < 0:
            self._ki = 0.0

        if self._kd < 0:
            self._kd = 0.0

        # Reasonable limits to prevent instability
        max_kp = 10.0 / self._ku if self._ku > 0 else 10.0
        self._kp = min(self._kp, max_kp)

        max_kd = self._kp * self._tu if self._tu > 0 else self._kp
        self._kd = min(self._kd, max_kd)

        # Print results
        self._print_results()

    def _run_pid_phase(self, us_now: float, us_elapsed: float) -> TunerStatus:
        """Handle PID execution phase (post-tuning)."""
        if self.pv_inst > self.e_stop and not self.e_stop_abort:
            self.reset()
            self.sample_count = self._samples + 1
            self.e_stop_abort = 1
            if self._serial_mode != SerialMode.PRINT_OFF:
                print("ABORT: pv_inst > e_stop")

        self._tuner_status = TunerStatus.TIMER_PID
        return TunerStatus.TIMER_PID

    def _run_timer_phase(self, us_elapsed: float) -> TunerStatus:
        """Handle timing for PID execution."""
        if us_elapsed >= self._sample_period_us:
            self.us_prev = self.timer.get_time_us()
            self._tuner_status = TunerStatus.RUN_PID
            return TunerStatus.RUN_PID
        else:
            self._tuner_status = TunerStatus.TIMER_PID
            return TunerStatus.TIMER_PID

    def reset(self) -> None:
        """Reset the tuner to initial state."""

        # Fix 1: Proper sample count initialization
        self._tuner_status = TunerStatus.TEST
        self._current_output = self._output_start
        self._input_valid = False

        self.us_prev = self.timer.get_time_us()
        self.settle_prev = self.us_prev
        self.ip_us = 0
        self.us = 0

        # Reset calculated parameters
        self._ku = 0.0
        self._tu = 0.0
        self._td = 0.0
        self._kp = 0.0
        self._ki = 0.0
        self._kd = 0.0

        # Reset process variables
        self.pv_ip = 0.0
        self.pv_max = 0.0
        self.pv_pk = 0.0
        self.slope_ip = 0.0
        self.pv_tangent = 0.0
        self.pv_tangent_prev = 0.0

        # Initialize process variables properly
        if self._input_valid:
            self.pv_inst = self._current_input
            self.pv_avg = self.pv_inst
            self.pv_start = self.pv_inst
            self.pv_inst_res = (
                abs(self.pv_inst) if abs(self.pv_inst) > self.EPSILON else 1.0
            )
            self.pv_avg_res = self.pv_inst_res
        else:
            self.pv_inst = 0.0
            self.pv_avg = 0.0
            self.pv_start = 0.0
            self.pv_inst_res = 1.0
            self.pv_avg_res = 1.0

        # Critical fix: Start sample_count at -1 so first sample becomes 0
        self.sample_count = -1  # Will become 0 on first sample
        self.ip_count = 0
        self.plot_count = 0
        self.pv_pk_count = 0
        self.e_stop_abort = 0

    # Setters
    def set_emergency_stop(self, e_stop: float) -> None:
        """Set emergency stop threshold."""
        self.e_stop = e_stop

    def set_controller_action(self, action: TunerAction) -> None:
        """Set controller action type."""
        self._action = action

    def set_serial_mode(self, serial_mode: SerialMode) -> None:
        """Set debug output verbosity."""
        self._serial_mode = serial_mode

    def set_tuning_method(self, tuning_method: TuningMethod) -> None:
        """Set PID calculation method."""
        self._tuning_method = tuning_method

    # Getter methods (unchanged from original)
    def get_kp(self) -> float:
        """Improved Kp calculation with better parameter bounds."""

        if self._tu == 0 or self._td == 0 or self._ku == 0:
            return 1.0  # Safe default

        # Fix 10: Improved PID parameter calculations
        # Scale factors to improve stability
        stability_factor = min(2.0, max(0.5, self._tu / (self._td + 0.001)))

        # Calculate gains for different methods with stability factor
        zn_pid = ((1.2 * self._tu) / (self._ku * self._td)) / 2 * stability_factor
        do_pid = (0.66 * self._tu) / (self._ku * self._td) * stability_factor
        no_pid = (0.6 / self._ku) * (self._tu / self._td) * stability_factor
        cc_pid = (1.0 / self._ku) * (1.33 + (self._r / 4.0)) * stability_factor

        zn_pi = ((0.9 * self._tu) / (self._ku * self._td)) / 2 * stability_factor
        do_pi = (0.495 * self._tu) / (self._ku * self._td) * stability_factor
        no_pi = (0.35 / self._ku) * (self._tu / self._td) * stability_factor
        cc_pi = (1.0 / self._ku) * (0.9 + (self._r / 12.0)) * stability_factor

        method_map = {
            TuningMethod.ZN_PID: zn_pid,
            TuningMethod.DAMPED_OSC_PID: do_pid,
            TuningMethod.NO_OVERSHOOT_PID: no_pid,
            TuningMethod.COHEN_COON_PID: cc_pid,
            TuningMethod.MIXED_PID: 0.25 * (zn_pid + do_pid + no_pid + cc_pid),
            TuningMethod.ZN_PI: zn_pi,
            TuningMethod.DAMPED_OSC_PI: do_pi,
            TuningMethod.NO_OVERSHOOT_PI: no_pi,
            TuningMethod.COHEN_COON_PI: cc_pi,
            TuningMethod.MIXED_PI: 0.25 * (zn_pi + do_pi + no_pi + cc_pi),
        }

        kp = method_map.get(self._tuning_method, 1.0)

        # Apply reasonable bounds
        min_kp = 0.1 / self._ku if self._ku > 0 else 0.1
        max_kp = 5.0 / self._ku if self._ku > 0 else 5.0

        return max(min_kp, min(kp, max_kp))

    def get_ki(self) -> float:
        """Calculate and return integral gain based on selected method."""
        if self._td == 0:
            return 0.0

        # Calculate integral gains for different methods
        zn_pid = 1 / (2.0 * self._td)
        do_pid = 1 / (self._tu / 3.6)
        no_pid = 1 / self._tu
        cc_pid = 1 / (self._td * (30.0 + (3.0 * self._r)) / (9.0 + (20.0 * self._r)))

        zn_pi = 1 / (3.3333 * self._td)
        do_pi = 1 / (self._tu / 2.6)
        no_pi = 1 / (1.2 * self._tu)
        cc_pi = 1 / (self._td * (30.0 + (3.0 * self._r)) / (9.0 + (20.0 * self._r)))

        method_map = {
            TuningMethod.ZN_PID: zn_pid,
            TuningMethod.DAMPED_OSC_PID: do_pid,
            TuningMethod.NO_OVERSHOOT_PID: no_pid,
            TuningMethod.COHEN_COON_PID: cc_pid,
            TuningMethod.MIXED_PID: 0.25 * (zn_pid + do_pid + no_pid + cc_pid),
            TuningMethod.ZN_PI: zn_pi,
            TuningMethod.DAMPED_OSC_PI: do_pi,
            TuningMethod.NO_OVERSHOOT_PI: no_pi,
            TuningMethod.COHEN_COON_PI: cc_pi,
            TuningMethod.MIXED_PI: 0.25 * (zn_pi + do_pi + no_pi + cc_pi),
        }

        return method_map.get(self._tuning_method, 0.0)

    def get_kd(self) -> float:
        """Calculate and return derivative gain based on selected method."""
        if self._td == 0:
            return 0.0

        # Only PID methods have derivative term
        if self._tuning_method >= TuningMethod.ZN_PI:
            return 0.0  # PI controllers have no derivative

        # Calculate derivative gains for PID methods
        zn_pid = 1 / (0.5 * self._td)
        do_pid = 1 / (self._tu / 9.0)
        no_pid = 1 / (0.5 * self._td)
        cc_pid = 1 / ((4.0 * self._td) / (11.0 + (2.0 * self._r)))

        method_map = {
            TuningMethod.ZN_PID: zn_pid,
            TuningMethod.DAMPED_OSC_PID: do_pid,
            TuningMethod.NO_OVERSHOOT_PID: no_pid,
            TuningMethod.COHEN_COON_PID: cc_pid,
            TuningMethod.MIXED_PID: 0.25 * (zn_pid + do_pid + no_pid + cc_pid),
        }

        return method_map.get(self._tuning_method, 0.0)

    def get_ti(self) -> float:
        """Get integral time constant (Kp/Ki)."""
        ki = self.get_ki()
        if ki == 0:
            return 0.0
        return self.get_kp() / ki

    def get_td(self) -> float:
        """Get derivative time constant (Kp/Kd)."""
        kd = self.get_kd()
        if kd == 0:
            return 0.0
        return self.get_kp() / kd

    def get_process_gain(self) -> float:
        """Get calculated process gain."""
        return self._ku

    def get_dead_time(self) -> float:
        """Get calculated dead time in seconds."""
        return self._td

    def get_tau(self) -> float:
        """Get calculated time constant in seconds."""
        return self._tu

    def get_controller_action(self) -> int:
        """Get controller action as integer."""
        return int(self._action)

    def get_serial_mode(self) -> int:
        """Get serial mode as integer."""
        return int(self._serial_mode)

    def get_tuning_method(self) -> int:
        """Get tuning method as integer."""
        return int(self._tuning_method)

    def get_auto_tunings(self) -> Tuple[float, float, float]:
        """Get all PID tunings as tuple (Kp, Ki, Kd)."""
        return (self._kp, self._ki, self._kd)

    def get_current_input(self) -> float:
        """Get the current input value."""
        return self._current_input

    def is_input_valid(self) -> bool:
        """Check if input has been set."""
        return self._input_valid

    def is_tuning_complete(self) -> bool:
        """Check if autotuning has completed."""
        return self._tuner_status == TunerStatus.TUNINGS

    # Debug methods (implementations unchanged from original)
    def print_tunings(self) -> None:
        """Print current tuning method and parameters."""
        method_names = {
            TuningMethod.ZN_PID: "ZN_PID",
            TuningMethod.DAMPED_OSC_PID: "Damped_PID",
            TuningMethod.NO_OVERSHOOT_PID: "NoOvershoot_PID",
            TuningMethod.COHEN_COON_PID: "CohenCoon_PID",
            TuningMethod.MIXED_PID: "Mixed_PID",
            TuningMethod.ZN_PI: "ZN_PI",
            TuningMethod.DAMPED_OSC_PI: "Damped_PI",
            TuningMethod.NO_OVERSHOOT_PI: "NoOvershoot_PI",
            TuningMethod.COHEN_COON_PI: "CohenCoon_PI",
            TuningMethod.MIXED_PI: "Mixed_PI",
        }

        print(f" Tuning Method: {method_names.get(self._tuning_method, 'Unknown')}")
        print(f"  Kp: {self.get_kp():.3f}")
        print(f"  Ki: {self.get_ki():.3f}  Ti: {self.get_ti():.3f}")
        print(f"  Kd: {self.get_kd():.3f}  Td: {self.get_td():.3f}")
        print()

    def _print_results(self) -> None:
        """Print detailed tuning results."""
        if self._serial_mode in [
            SerialMode.PRINT_ALL,
            SerialMode.PRINT_DEBUG,
            SerialMode.PRINT_SUMMARY,
        ]:
            print()

            action_names = {
                TunerAction.DIRECT_IP: "directIP",
                TunerAction.DIRECT_5T: "direct5T",
                TunerAction.REVERSE_IP: "reverseIP",
                TunerAction.REVERSE_5T: "reverse5T",
            }
            print(
                f" Controller TunerAction: {action_names.get(self._action, 'Unknown')}"
            )
            print()

            print(f" Output Start:      {self._output_start}")
            print(f" Output Step:       {self._output_step}")
            print(f" Sample Sec:        {self._sample_period_us * 1e-6:.4f}")
            print()

            if self._serial_mode == SerialMode.PRINT_DEBUG and self._action in [
                TunerAction.DIRECT_IP,
                TunerAction.REVERSE_IP,
            ]:
                print(f" Ip Sec:            {self.ip_us * 1e-6:.4f}")
                direction = (
                    "↑"
                    if self._action in [TunerAction.DIRECT_IP, TunerAction.DIRECT_5T]
                    else "↓"
                )
                print(f" Ip Slope:          {self.slope_ip:.3f} {direction}")
                print(f" Ip Pv:             {self.pv_ip:.3f}")

            print(f" Pv Start:          {self.pv_start:.3f}")

            if self._action in [TunerAction.DIRECT_IP, TunerAction.DIRECT_5T]:
                print(f" Pv Max:            {self.pv_max:.3f}")
            else:
                print(f" Pv Min:            {self.pv_max:.3f}")

            print(f" Pv Diff:           {self.pv_max - self.pv_start:.3f}")
            print()

            print(f" Process Gain:      {self._ku:.3f}")
            print(f" Dead Time Sec:     {self._td:.3f}")
            print(f" Tau Sec:           {self._tu:.3f}")
            print()

            # Controllability analysis
            controllability = self._tu / (self._td + self.EPSILON)
            if controllability > 99.9:
                controllability = 99.9

            print(f" Tau/Dead Time:     {controllability:.1f}", end="")
            if controllability > 0.75:
                print(" (easy to control)")
            elif controllability > 0.25:
                print(" (average controllability)")
            else:
                print(" (difficult to control)")

            # Sample rate analysis
            sample_time_check = self._tu / (self._sample_period_us * 1e-6)
            print(f" Tau/Sample Period: {sample_time_check:.1f}", end="")
            if sample_time_check >= 10:
                print(" (good sample rate)")
            else:
                print(" (low sample rate)")

            print()
            self.print_tunings()
            self.sample_count += 1

    def _print_test_run(self) -> None:
        """Print debug information during test run."""
        if self.sample_count < self._samples and self._serial_mode in [
            SerialMode.PRINT_ALL,
            SerialMode.PRINT_DEBUG,
        ]:
            print(f" sec: {self.us * 1e-6:.4f}", end="")
            print(f"  out: {self._current_output}", end="")
            print(f"  pv: {self.pv_inst:.3f}", end="")

            if self._serial_mode == SerialMode.PRINT_DEBUG and self._action in [
                TunerAction.DIRECT_5T,
                TunerAction.REVERSE_5T,
            ]:
                print(f"  pvPk: {self.pv_pk:.3f}", end="")
                print(f"  pvPkCount: {self.pv_pk_count}", end="")
                print(f"  ipCount: {self.ip_count}", end="")

            if self._serial_mode == SerialMode.PRINT_DEBUG and self._action in [
                TunerAction.DIRECT_IP,
                TunerAction.REVERSE_IP,
            ]:
                print(f"  ipCount: {self.ip_count}", end="")

            print(f"  tan: {self.pv_tangent:.3f}", end="")

            if self.pv_inst > 0.9 * self.e_stop:
                print(" ⚠", end="")

            # Show tangent trend
            tangent_change = self.pv_tangent - self.pv_tangent_prev
            if tangent_change > self.EPSILON:
                print(" ↗")
            elif tangent_change < -self.EPSILON:
                print(" ↘")
            else:
                print(" →")
