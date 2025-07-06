"""
AutoAdamPID - Self-Tuning PID Controller with Instance Method Interface

This module provides an auto-tuning PID controller that:
- Uses instance methods for input/output instead of callable functions
- Automatically tunes itself using STune on first run or when configured
- Persists settings in YAML configuration files
- Logs PID performance data to SQLite database
- Maintains all AdamPID functionality with seamless interface
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from loguru import logger

from .adampid import AdamPID, Action, Control, PMode, DMode, IAwMode
from .s_tune import STune, TuningMethod, TunerAction, TunerStatus, SerialMode
from .timing_base import TimerBase
from .real_time_timer import RealTimeTimer
from .exceptions import AdamPIDError, TuningError


class AutoAdamPIDError(AdamPIDError):
    """Exception raised by AutoAdamPID operations."""

    pass


class AutoAdamPID:
    """
    Self-Tuning PID Controller with Instance Method Interface and Persistent Configuration.

    This class combines AdamPID control with STune auto-tuning capabilities,
    automatic configuration persistence, and comprehensive data logging using
    instance methods for complete application control over timing and data flow.

    Features:
    - Instance method interface for input/output operations
    - Automatic PID tuning using STune inflection point method
    - YAML-based configuration persistence
    - SQLite database logging with configurable intervals
    - Seamless AdamPID interface delegation
    - Comprehensive error handling and logging

    Usage:
        auto_pid = AutoAdamPID("config.yaml")

        # In control loop:
        auto_pid.set_input(sensor_reading)
        auto_pid.set_setpoint(desired_value)
        if auto_pid.compute():
            output = auto_pid.get_output()
            apply_output_to_actuator(output)
    """

    def __init__(
        self,
        config_path: str,
        timer: Optional[TimerBase] = None,
    ):
        """
        Initialize AutoAdamPID with configuration file.

        Args:
            config_path: Path to YAML configuration file
            timer: Timer implementation for consistent timing
        """
        self.config_path = Path(config_path)
        self.timer = timer or RealTimeTimer()

        # Internal components
        self._pid: Optional[AdamPID] = None
        self._s_tune: Optional[STune] = None

        # Configuration and state
        self.config: Dict[str, Any] = {}
        self._db_connection: Optional[sqlite3.Connection] = None
        self._last_log_time: float = 0.0
        self._auto_tuning_complete: bool = False
        self._tuning_in_progress: bool = False

        # Current values managed by this class
        self._current_input: float = 0.0
        self._current_setpoint: float = 0.0
        self._current_output: float = 0.0
        self._input_valid: bool = False
        self._setpoint_valid: bool = False

        # Initialize logger
        logger.info(f"Initializing AutoAdamPID with config: {self.config_path}")

        # Load configuration and setup
        self._load_config()
        self._setup_database()
        self._create_pid_controller()

        # Run auto-tuning if enabled and needed
        if self._should_auto_tune():
            logger.info(
                "Auto-tuning enabled and required - will start when input/setpoint provided"
            )
        else:
            logger.info("Auto-tuning not required - using existing PID parameters")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup()

    def set_input(self, input_value: float) -> None:
        """
        Set the current process variable (sensor reading).

        Args:
            input_value: Current measured process variable
        """
        self._current_input = input_value
        self._input_valid = True

        # Pass to internal PID controller if available and not tuning
        if self._pid and not self._tuning_in_progress:
            self._pid.set_input(input_value)

        # CRITICAL FIX: Always pass to tuner if tuning is in progress and tuner exists
        if self._s_tune and self._tuning_in_progress:
            self._s_tune.set_input(input_value)

    def set_setpoint(self, setpoint_value: float) -> None:
        """
        Set the current setpoint (desired value).

        Args:
            setpoint_value: Desired process variable value
        """
        self._current_setpoint = setpoint_value
        self._setpoint_valid = True

        # Pass to internal PID controller if available and not tuning
        if self._pid and not self._tuning_in_progress:
            self._pid.set_setpoint(setpoint_value)

    def get_output(self) -> float:
        """
        Get the current controller output.

        Returns:
            Current controller output value (from PID or tuner)
        """
        if self._tuning_in_progress and self._s_tune:
            return self._s_tune.get_output()
        elif self._pid:
            return self._pid.get_output()
        else:
            return self._current_output

    def compute(self) -> bool:
        """
        Perform PID calculation or auto-tuning step and log data.

        This method handles both normal PID operation and auto-tuning.
        During auto-tuning, it runs the tuner instead of the PID controller.

        Returns:
            True if calculation was performed, False if not time yet or not ready
        """
        if not self._input_valid:
            logger.warning("Input must be set before calling compute()")
            return False

        # CRITICAL FIX: Check if we should start auto-tuning BEFORE running tuning step
        should_start_tuning = (
            self._should_auto_tune()
            and not self._tuning_in_progress
            and not self._auto_tuning_complete
            and self._setpoint_valid  # Need setpoint to know where we're going
        )

        if should_start_tuning:
            logger.info("Starting auto-tuning process")
            self._start_auto_tuning()
            # After starting tuning, ensure input is set again for immediate use
            if self._s_tune:
                self._s_tune.set_input(self._current_input)

        # Handle auto-tuning process
        if self._tuning_in_progress:
            return self._run_tuning_step()

        # Handle normal PID operation
        if self._pid:
            if not self._setpoint_valid:
                logger.warning("Setpoint must be set before calling compute()")
                return False

            # Perform PID calculation
            result = self._pid.compute()

            # Update current output and log if calculation was performed
            if result:
                self._current_output = self._pid.get_output()
                self._log_to_database()

            return result

        logger.error("Neither PID controller nor tuner is available")
        return False

    def _start_auto_tuning(self) -> None:
        """Initialize and start the auto-tuning process."""
        try:
            auto_tune_config = self.config["auto_tune"]

            # Parse tuning method and action from config
            tuning_method = TuningMethod[auto_tune_config["tuning_method"]]
            action = TunerAction[auto_tune_config["action"]]

            # Create STune instance
            self._s_tune = STune(
                tuning_method=tuning_method,
                action=action,
                serial_mode=SerialMode.PRINT_SUMMARY,
                timer=self.timer,
            )

            # Configure STune parameters
            self._s_tune.configure(
                input_span=auto_tune_config["input_span"],
                output_span=auto_tune_config["output_span"],
                output_start=auto_tune_config["output_start"],
                output_step=auto_tune_config["output_step"],
                test_time_sec=auto_tune_config["test_time_sec"],
                settle_time_sec=auto_tune_config["settle_time_sec"],
                samples=auto_tune_config["samples"],
            )

            # Set emergency stop
            self._s_tune.set_emergency_stop(auto_tune_config["emergency_stop"])

            # CRITICAL FIX: Set current input to STune immediately after configuration
            if self._input_valid:
                self._s_tune.set_input(self._current_input)
                logger.info(
                    f"STune input initialized with value: {self._current_input}"
                )
            else:
                logger.warning("STune created but no input value available yet")

            # Mark tuning as in progress
            self._tuning_in_progress = True
            self._tuning_start_time = self.timer.get_time_s()

            logger.info("STune configured and auto-tuning started")

        except Exception as e:
            logger.error(f"Failed to start auto-tuning: {e}")
            self._tuning_in_progress = False
            raise AutoAdamPIDError(f"Auto-tuning initialization failed: {e}")

    def _run_tuning_step(self) -> bool:
        """Execute one step of the auto-tuning process."""
        if not self._s_tune:
            logger.error("Tuner not initialized")
            self._tuning_in_progress = False
            return False

        try:
            # Check timeout
            current_time = self.timer.get_time_s()
            max_duration = self.config["auto_tune"]["max_duration_sec"]

            if current_time - self._tuning_start_time > max_duration:
                logger.error(f"Auto-tuning timeout after {max_duration}s")
                self._tuning_in_progress = False
                raise AutoAdamPIDError("Auto-tuning timeout")

            # CRITICAL FIX: Ensure input is always set before calling run()
            if self._input_valid:
                self._s_tune.set_input(self._current_input)
            else:
                logger.error("No valid input available for tuning step")
                return False

            # Run one tuning step
            status = self._s_tune.run()

            # Update current output from tuner
            self._current_output = self._s_tune.get_output()

            # Check if tuning completed
            if status == TunerStatus.TUNINGS:
                self._complete_auto_tuning()
                return True

            # Log tuning progress if needed
            self._log_tuning_progress(status)

            return True

        except TuningError as e:
            logger.error(f"Auto-tuning failed: {e}")
            self._tuning_in_progress = False
            raise AutoAdamPIDError(f"Auto-tuning failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during auto-tuning: {e}")
            self._tuning_in_progress = False
            raise AutoAdamPIDError(f"Auto-tuning error: {e}")

    def _complete_auto_tuning(self) -> None:
        """Complete the auto-tuning process and update PID controller."""
        if not self._s_tune:
            return

        try:
            # Extract tuned parameters
            kp, ki, kd = self._s_tune.get_auto_tunings()

            logger.info(
                f"Auto-tuned PID parameters: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}"
            )

            # Update configuration
            self.config["pid"]["kp"] = float(kp)
            self.config["pid"]["ki"] = float(ki)
            self.config["pid"]["kd"] = float(kd)

            # Update PID controller with new parameters
            if self._pid:
                self._pid.set_tunings(kp, ki, kd)
                # Set current values to PID controller for smooth transition
                if self._input_valid:
                    self._pid.set_input(self._current_input)
                if self._setpoint_valid:
                    self._pid.set_setpoint(self._current_setpoint)

            # Save updated configuration
            self._save_config()

            # Mark tuning complete
            self._tuning_in_progress = False
            self._auto_tuning_complete = True

            logger.info(
                "Auto-tuning process completed successfully and configuration saved"
            )

        except Exception as e:
            logger.error(f"Failed to complete auto-tuning: {e}")
            self._tuning_in_progress = False
            raise AutoAdamPIDError(f"Auto-tuning completion failed: {e}")

    def _log_tuning_progress(self, status: TunerStatus) -> None:
        """Log auto-tuning progress if database logging is enabled."""
        if not self._db_connection:
            return

        try:
            current_time = self.timer.get_time_s()

            # Log tuning data to a special tuning table
            self._db_connection.execute("""
                CREATE TABLE IF NOT EXISTS tuning_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL,
                    input_value REAL NOT NULL,
                    output_value REAL NOT NULL
                )
            """)

            self._db_connection.execute(
                """
                INSERT INTO tuning_logs (timestamp, status, input_value, output_value)
                VALUES (?, ?, ?, ?)
            """,
                (current_time, status.name, self._current_input, self._current_output),
            )

            self._db_connection.commit()

        except Exception as e:
            logger.warning(f"Failed to log tuning progress: {e}")

    def _load_config(self) -> None:
        """Load configuration from YAML file or create default."""
        try:
            if self.config_path.exists():
                logger.info(f"Loading existing configuration from {self.config_path}")
                with open(self.config_path, "r") as f:
                    self.config = yaml.safe_load(f)
                self._validate_config()
            else:
                logger.warning(
                    f"Configuration file not found, creating default: {self.config_path}"
                )
                self._create_default_config()
                self._save_config()

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise AutoAdamPIDError(f"Configuration parsing error: {e}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise AutoAdamPIDError(f"Configuration loading error: {e}")

    def _create_default_config(self) -> None:
        """Create default configuration structure."""
        self.config = {
            "pid": {
                "kp": 0.0,
                "ki": 0.0,
                "kd": 0.0,
                "action": "DIRECT",
                "p_mode": "P_ON_ERROR",
                "d_mode": "D_ON_MEAS",
                "i_aw_mode": "I_AW_CONDITION",
                "sample_time_us": 100000,
                "output_limits": {"min": -100.0, "max": 100.0},
            },
            "auto_tune": {
                "enabled": True,
                "tuning_method": "COHEN_COON_PID",
                "action": "DIRECT_IP",
                "input_span": 100.0,
                "output_span": 200.0,
                "output_start": 0.0,
                "output_step": 30.0,
                "test_time_sec": 60,
                "settle_time_sec": 10,
                "samples": 600,
                "emergency_stop": 90.0,
                "max_duration_sec": 120,
                "convergence_tolerance": 0.01,
            },
            "logging": {
                "database_path": "pid_data.db",
                "interval_sec": 1.0,
                "enabled": True,
            },
        }

    def _validate_config(self) -> None:
        """Validate loaded configuration structure."""
        required_sections = ["pid", "auto_tune", "logging"]
        for section in required_sections:
            if section not in self.config:
                raise AutoAdamPIDError(
                    f"Missing required configuration section: {section}"
                )

        # Validate PID section
        pid_required = ["kp", "ki", "kd", "action", "sample_time_us", "output_limits"]
        for key in pid_required:
            if key not in self.config["pid"]:
                raise AutoAdamPIDError(f"Missing required PID parameter: {key}")

        # Validate output limits
        if (
            "min" not in self.config["pid"]["output_limits"]
            or "max" not in self.config["pid"]["output_limits"]
        ):
            raise AutoAdamPIDError("Missing output limits min/max values")

    def _save_config(self) -> None:
        """Save current configuration to YAML file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise AutoAdamPIDError(f"Configuration save error: {e}")

    def _setup_database(self) -> None:
        """Setup SQLite database for logging."""
        if not self.config["logging"].get("enabled", True):
            logger.info("Database logging disabled in configuration")
            return

        try:
            db_path = Path(self.config["logging"]["database_path"])

            # Create directory if needed
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if database exists
            db_exists = db_path.exists()
            if not db_exists:
                logger.info(f"Creating new database: {db_path}")

            # Connect and create table if needed
            self._db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db_connection.execute("""
                CREATE TABLE IF NOT EXISTS pid_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    setpoint REAL NOT NULL,
                    process_variable REAL NOT NULL,
                    control_output REAL NOT NULL,
                    error REAL NOT NULL,
                    p_term REAL NOT NULL,
                    i_term REAL NOT NULL,
                    d_term REAL NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'PID'
                )
            """)
            self._db_connection.commit()

            logger.info(f"Database setup completed: {db_path}")

        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            # Don't fail initialization for database errors
            self._db_connection = None

    def _create_pid_controller(self) -> None:
        """Create internal PID controller from configuration."""
        try:
            pid_config = self.config["pid"]

            # Parse enums from strings
            action = (
                Action.DIRECT if pid_config["action"] == "DIRECT" else Action.REVERSE
            )
            p_mode = PMode[pid_config.get("p_mode", "P_ON_ERROR")]
            d_mode = DMode[pid_config.get("d_mode", "D_ON_MEAS")]
            i_aw_mode = IAwMode[pid_config.get("i_aw_mode", "I_AW_CONDITION")]

            # Create PID controller
            self._pid = AdamPID(
                kp=pid_config["kp"],
                ki=pid_config["ki"],
                kd=pid_config["kd"],
                p_mode=p_mode,
                d_mode=d_mode,
                i_aw_mode=i_aw_mode,
                action=action,
                timer=self.timer,
            )

            # Set additional parameters
            self._pid.set_sample_time_us(pid_config["sample_time_us"])
            self._pid.set_output_limits(
                pid_config["output_limits"]["min"], pid_config["output_limits"]["max"]
            )

            # Set to automatic mode
            self._pid.set_mode(Control.AUTOMATIC)

            logger.info("PID controller created and configured")

        except Exception as e:
            logger.error(f"Failed to create PID controller: {e}")
            raise AutoAdamPIDError(f"PID controller creation failed: {e}")

    def _should_auto_tune(self) -> bool:
        """Determine if auto-tuning should be performed."""
        auto_tune_config = self.config["auto_tune"]

        # Check if auto-tuning is enabled
        if not auto_tune_config.get("enabled", False):
            return False

        # Check if PID parameters are zero (indicating need for tuning)
        pid_config = self.config["pid"]
        if pid_config["kp"] == 0 and pid_config["ki"] == 0 and pid_config["kd"] == 0:
            return True

        # Could add other conditions here (e.g., poor performance detection)
        return False

    def _log_to_database(self) -> None:
        """Log current PID state to database."""
        if not self._db_connection or not self._pid:
            return

        current_time = self.timer.get_time_s()
        log_interval = self.config["logging"]["interval_sec"]

        # Check if it's time to log
        if current_time - self._last_log_time < log_interval:
            return

        try:
            pv = self._current_input
            sp = self._current_setpoint
            error = sp - pv

            # Get PID terms
            p_term = self._pid.get_p_term()
            i_term = self._pid.get_i_term()
            d_term = self._pid.get_d_term()
            output = self._current_output

            # Determine mode
            mode = "TUNING" if self._tuning_in_progress else "PID"

            # Insert into database
            self._db_connection.execute(
                """
                INSERT INTO pid_logs 
                (timestamp, setpoint, process_variable, control_output, error, p_term, i_term, d_term, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (current_time, sp, pv, output, error, p_term, i_term, d_term, mode),
            )

            self._db_connection.commit()
            self._last_log_time = current_time

        except Exception as e:
            logger.warning(f"Failed to log to database: {e}")

    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._db_connection:
            try:
                self._db_connection.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self._db_connection = None

    # Public API Methods - AdamPID Interface Delegation

    def set_tunings(
        self,
        kp: float,
        ki: float,
        kd: float,
        p_mode: Optional[PMode] = None,
        d_mode: Optional[DMode] = None,
        i_aw_mode: Optional[IAwMode] = None,
    ) -> None:
        """Set PID tuning parameters and update configuration."""
        if self._tuning_in_progress:
            logger.warning("Cannot change tunings while auto-tuning is in progress")
            return

        if not self._pid:
            logger.error("PID controller not initialized")
            return

        self._pid.set_tunings(kp, ki, kd, p_mode, d_mode, i_aw_mode)

        # Update configuration
        self.config["pid"]["kp"] = float(kp)
        self.config["pid"]["ki"] = float(ki)
        self.config["pid"]["kd"] = float(kd)

        if p_mode is not None:
            self.config["pid"]["p_mode"] = p_mode.name
        if d_mode is not None:
            self.config["pid"]["d_mode"] = d_mode.name
        if i_aw_mode is not None:
            self.config["pid"]["i_aw_mode"] = i_aw_mode.name

        # Save configuration
        self._save_config()
        logger.info(f"PID tunings updated: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")

    def set_sample_time_us(self, new_sample_time_us: int) -> None:
        """Set sample time and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return

        self._pid.set_sample_time_us(new_sample_time_us)
        self.config["pid"]["sample_time_us"] = new_sample_time_us
        self._save_config()

    def set_output_limits(self, min_output: float, max_output: float) -> None:
        """Set output limits and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return

        self._pid.set_output_limits(min_output, max_output)
        self.config["pid"]["output_limits"]["min"] = float(min_output)
        self.config["pid"]["output_limits"]["max"] = float(max_output)
        self._save_config()

    def set_mode(self, mode: Control) -> None:
        """Set controller operation mode."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
        self._pid.set_mode(mode)

    def set_controller_direction(self, action: Action) -> None:
        """Set controller action and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return

        self._pid.set_controller_direction(action)
        self.config["pid"]["action"] = action.name
        self._save_config()

    def set_proportional_mode(self, p_mode: PMode) -> None:
        """Set proportional mode and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return

        self._pid.set_proportional_mode(p_mode)
        self.config["pid"]["p_mode"] = p_mode.name
        self._save_config()

    def set_derivative_mode(self, d_mode: DMode) -> None:
        """Set derivative mode and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return

        self._pid.set_derivative_mode(d_mode)
        self.config["pid"]["d_mode"] = d_mode.name
        self._save_config()

    def set_anti_windup_mode(self, i_aw_mode: IAwMode) -> None:
        """Set anti-windup mode and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return

        self._pid.set_anti_windup_mode(i_aw_mode)
        self.config["pid"]["i_aw_mode"] = i_aw_mode.name
        self._save_config()

    def set_output_sum(self, sum_value: float) -> None:
        """Set integral sum value directly."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
        self._pid.set_output_sum(sum_value)

    def initialize(self) -> None:
        """Initialize controller for bumpless transfer."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
        self._pid.initialize()

    def reset(self) -> None:
        """Reset all internal state variables."""
        if self._pid:
            self._pid.reset()

        # Reset internal state
        self._current_output = 0.0
        self._input_valid = False
        self._setpoint_valid = False

        # Stop any tuning in progress
        if self._tuning_in_progress:
            self._tuning_in_progress = False
            logger.info("Auto-tuning stopped due to reset")

    # Query methods - delegate to internal PID

    def get_kp(self) -> float:
        """Get proportional gain."""
        return self._pid.get_kp() if self._pid else 0.0

    def get_ki(self) -> float:
        """Get integral gain."""
        return self._pid.get_ki() if self._pid else 0.0

    def get_kd(self) -> float:
        """Get derivative gain."""
        return self._pid.get_kd() if self._pid else 0.0

    def get_p_term(self) -> float:
        """Get proportional term component."""
        return self._pid.get_p_term() if self._pid else 0.0

    def get_i_term(self) -> float:
        """Get integral term component."""
        return self._pid.get_i_term() if self._pid else 0.0

    def get_d_term(self) -> float:
        """Get derivative term component."""
        return self._pid.get_d_term() if self._pid else 0.0

    def get_output_sum(self) -> float:
        """Get integral sum value."""
        return self._pid.get_output_sum() if self._pid else 0.0

    def get_mode(self) -> int:
        """Get current operation mode as integer."""
        return self._pid.get_mode() if self._pid else 0

    def get_direction(self) -> int:
        """Get controller action as integer."""
        return self._pid.get_direction() if self._pid else 0

    def get_p_mode(self) -> int:
        """Get proportional mode as integer."""
        return self._pid.get_p_mode() if self._pid else 0

    def get_d_mode(self) -> int:
        """Get derivative mode as integer."""
        return self._pid.get_d_mode() if self._pid else 0

    def get_aw_mode(self) -> int:
        """Get anti-windup mode as integer."""
        return self._pid.get_aw_mode() if self._pid else 0

    def get_sample_time_us(self) -> int:
        """Get sample time in microseconds."""
        return self._pid.get_sample_time_us() if self._pid else 0

    def get_output_limits(self) -> tuple[float, float]:
        """Get output limits as (min, max) tuple."""
        return self._pid.get_output_limits() if self._pid else (0.0, 0.0)

    def get_current_input(self) -> float:
        """Get the current input value."""
        return self._current_input

    def get_current_setpoint(self) -> float:
        """Get the current setpoint value."""
        return self._current_setpoint

    def get_current_error(self) -> float:
        """Get the current error value."""
        if self._pid:
            return self._pid.get_current_error()
        return self._current_setpoint - self._current_input

    def is_input_valid(self) -> bool:
        """Check if input has been set."""
        return self._input_valid

    def is_setpoint_valid(self) -> bool:
        """Check if setpoint has been set."""
        return self._setpoint_valid

    # Additional AutoAdamPID specific methods

    def get_auto_tuning_results(self) -> Dict[str, float]:
        """Get auto-tuning results if available."""
        if not self._s_tune:
            return {}

        return {
            "process_gain": self._s_tune.get_process_gain(),
            "dead_time": self._s_tune.get_dead_time(),
            "time_constant": self._s_tune.get_tau(),
            "kp": self._s_tune.get_kp(),
            "ki": self._s_tune.get_ki(),
            "kd": self._s_tune.get_kd(),
        }

    def is_auto_tuning_complete(self) -> bool:
        """Check if auto-tuning has completed."""
        return self._auto_tuning_complete

    def is_auto_tuning_in_progress(self) -> bool:
        """Check if auto-tuning is currently running."""
        return self._tuning_in_progress

    def stop_auto_tuning(self) -> None:
        """Stop auto-tuning if it's in progress."""
        if self._tuning_in_progress:
            self._tuning_in_progress = False
            logger.info("Auto-tuning stopped by user request")

    def force_auto_tuning(self) -> None:
        """Force auto-tuning to start on next compute() call."""
        if self._tuning_in_progress:
            logger.warning("Auto-tuning already in progress")
            return

        # Mark that tuning should start
        self._auto_tuning_complete = False
        logger.info("Auto-tuning will start on next compute() call")

    def force_config_save(self) -> None:
        """Force save current configuration to file."""
        self._save_config()

    def get_database_path(self) -> str:
        """Get path to logging database."""
        return self.config["logging"]["database_path"]

    def get_config(self) -> Dict[str, Any]:
        """Get a copy of the current configuration."""
        return self.config.copy()

    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""

        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        update_nested_dict(self.config, config_updates)
        self._save_config()
        logger.info("Configuration updated")

    def get_tuning_progress(self) -> Dict[str, Any]:
        """Get current auto-tuning progress information."""
        if not self._tuning_in_progress or not self._s_tune:
            return {
                "in_progress": False,
                "elapsed_time": 0.0,
                "max_duration": 0.0,
                "progress_percent": 0.0,
            }

        current_time = self.timer.get_time_s()
        elapsed_time = current_time - self._tuning_start_time
        max_duration = self.config["auto_tune"]["max_duration_sec"]
        progress_percent = (elapsed_time / max_duration) * 100.0

        return {
            "in_progress": True,
            "elapsed_time": elapsed_time,
            "max_duration": max_duration,
            "progress_percent": min(progress_percent, 100.0),
            "current_input": self._current_input,
            "current_output": self._current_output,
        }

    def enable_database_logging(self, enabled: bool = True) -> None:
        """Enable or disable database logging."""
        self.config["logging"]["enabled"] = enabled

        if enabled and not self._db_connection:
            self._setup_database()
        elif not enabled and self._db_connection:
            self._db_connection.close()
            self._db_connection = None

        self._save_config()
        logger.info(f"Database logging {'enabled' if enabled else 'disabled'}")

    def set_log_interval(self, interval_sec: float) -> None:
        """Set the database logging interval in seconds."""
        self.config["logging"]["interval_sec"] = interval_sec
        self._save_config()
        logger.info(f"Log interval set to {interval_sec}s")

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "tuning" if self._tuning_in_progress else "ready"
        if self._pid:
            return (
                f"AutoAdamPID(config={self.config_path}, "
                f"Kp={self.get_kp():.3f}, Ki={self.get_ki():.3f}, "
                f"Kd={self.get_kd():.3f}, status={status}, "
                f"auto_tuned={self._auto_tuning_complete})"
            )
        else:
            return f"AutoAdamPID(config={self.config_path}, not_initialized, status={status})"
