"""
AutoAdamPID - Self-Tuning PID Controller with Persistent Configuration and Logging

This module provides an auto-tuning PID controller that:
- Automatically tunes itself using STune on first run or when configured
- Persists settings in YAML configuration files
- Logs PID performance data to SQLite database
- Maintains all AdamPID functionality with seamless interface
"""

import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import yaml
from loguru import logger

from .adampid import AdamPID, Action, Control, PMode, DMode, IAwMode
from .s_tune import STune, TuningMethod, TunerAction, TunerStatus, SerialMode
from .timing_base import TimerBase
from .real_time_timer import RealTimeTimer
from .exceptions import AdamPIDError, ConfigurationError, TuningError


class AutoAdamPIDError(AdamPIDError):
    """Exception raised by AutoAdamPID operations."""
    pass


class AutoAdamPID:
    """
    Self-Tuning PID Controller with Persistent Configuration and Database Logging.
    
    This class combines AdamPID control with STune auto-tuning capabilities,
    automatic configuration persistence, and comprehensive data logging.
    
    Features:
    - Automatic PID tuning using STune inflection point method
    - YAML-based configuration persistence
    - SQLite database logging with configurable intervals
    - Seamless AdamPID interface delegation
    - Comprehensive error handling and logging
    """
    
    def __init__(
        self,
        config_path: str,
        input_var: Optional[Callable[[], float]] = None,
        output_var: Optional[Callable[[float], None]] = None,
        setpoint_var: Optional[Callable[[], float]] = None,
        timer: Optional[TimerBase] = None,
    ):
        """
        Initialize AutoAdamPID with configuration file and variable connections.
        
        Args:
            config_path: Path to YAML configuration file
            input_var: Function that returns current process variable
            output_var: Function that sets the control output
            setpoint_var: Function that returns current setpoint
            timer: Timer implementation for consistent timing
        """
        self.config_path = Path(config_path)
        self.timer = timer or RealTimeTimer()
        
        # Variable connections
        self._input_var = input_var
        self._output_var = output_var
        self._setpoint_var = setpoint_var
        
        # Internal components
        self._pid: Optional[AdamPID] = None
        self._s_tune: Optional[STune] = None
        
        # Configuration and state
        self.config: Dict[str, Any] = {}
        self._db_connection: Optional[sqlite3.Connection] = None
        self._last_log_time: float = 0.0
        self._auto_tuning_complete: bool = False
        
        # Initialize logger
        logger.info(f"Initializing AutoAdamPID with config: {self.config_path}")
        
        # Load configuration and setup
        self._load_config()
        self._setup_database()
        self._create_pid_controller()
        
        # Run auto-tuning if enabled and needed
        if self._should_auto_tune():
            logger.info("Auto-tuning enabled and required - starting auto-tuning process")
            self._run_auto_tuning()
        else:
            logger.info("Auto-tuning not required - using existing PID parameters")
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup()
        
    def _load_config(self) -> None:
        """Load configuration from YAML file or create default."""
        try:
            if self.config_path.exists():
                logger.info(f"Loading existing configuration from {self.config_path}")
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                self._validate_config()
            else:
                logger.warning(f"Configuration file not found, creating default: {self.config_path}")
                self._create_default_config()
                self._save_config()
                
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
            
    def _create_default_config(self) -> None:
        """Create default configuration structure."""
        self.config = {
            'pid': {
                'kp': 0.0,
                'ki': 0.0,
                'kd': 0.0,
                'action': 'DIRECT',
                'p_mode': 'P_ON_ERROR',
                'd_mode': 'D_ON_MEAS',
                'i_aw_mode': 'I_AW_CONDITION',
                'sample_time_us': 100000,
                'output_limits': {
                    'min': -100.0,
                    'max': 100.0
                }
            },
            'auto_tune': {
                'enabled': True,
                'tuning_method': 'COHEN_COON_PID',
                'action': 'DIRECT_IP',
                'input_span': 100.0,
                'output_span': 200.0,
                'output_start': 0.0,
                'output_step': 30.0,
                'test_time_sec': 60,
                'settle_time_sec': 10,
                'samples': 600,
                'emergency_stop': 90.0,
                'max_duration_sec': 120,
                'convergence_tolerance': 0.01
            },
            'logging': {
                'database_path': 'pid_data.db',
                'interval_sec': 1.0
            }
        }
        
    def _validate_config(self) -> None:
        """Validate loaded configuration structure."""
        required_sections = ['pid', 'auto_tune', 'logging']
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                sys.exit(1)
                
        # Validate PID section
        pid_required = ['kp', 'ki', 'kd', 'action', 'sample_time_us', 'output_limits']
        for key in pid_required:
            if key not in self.config['pid']:
                logger.error(f"Missing required PID parameter: {key}")
                sys.exit(1)
                
        # Validate output limits
        if 'min' not in self.config['pid']['output_limits'] or 'max' not in self.config['pid']['output_limits']:
            logger.error("Missing output limits min/max values")
            sys.exit(1)
            
    def _save_config(self) -> None:
        """Save current configuration to YAML file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            sys.exit(1)
            
    def _setup_database(self) -> None:
        """Setup SQLite database for logging."""
        try:
            db_path = Path(self.config['logging']['database_path'])
            
            # Create directory if needed
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if database exists
            db_exists = db_path.exists()
            if not db_exists:
                logger.warning(f"Database does not exist, creating: {db_path}")
                
            # Connect and create table if needed
            self._db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db_connection.execute('''
                CREATE TABLE IF NOT EXISTS pid_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    setpoint REAL NOT NULL,
                    process_variable REAL NOT NULL,
                    control_output REAL NOT NULL,
                    error REAL NOT NULL,
                    p_term REAL NOT NULL,
                    i_term REAL NOT NULL,
                    d_term REAL NOT NULL
                )
            ''')
            self._db_connection.commit()
            
            if not db_exists:
                logger.info(f"Database and table created successfully: {db_path}")
            else:
                logger.info(f"Connected to existing database: {db_path}")
                
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            sys.exit(1)
            
    def _create_pid_controller(self) -> None:
        """Create internal PID controller from configuration."""
        try:
            pid_config = self.config['pid']
            
            # Parse enums from strings
            action = Action.DIRECT if pid_config['action'] == 'DIRECT' else Action.REVERSE
            p_mode = PMode[pid_config.get('p_mode', 'P_ON_ERROR')]
            d_mode = DMode[pid_config.get('d_mode', 'D_ON_MEAS')]
            i_aw_mode = IAwMode[pid_config.get('i_aw_mode', 'I_AW_CONDITION')]
            
            # Create PID controller
            self._pid = AdamPID(
                input_var=self._input_var,
                output_var=self._output_var,
                setpoint_var=self._setpoint_var,
                kp=pid_config['kp'],
                ki=pid_config['ki'],
                kd=pid_config['kd'],
                p_mode=p_mode,
                d_mode=d_mode,
                i_aw_mode=i_aw_mode,
                action=action,
                timer=self.timer
            )
            
            # Set additional parameters
            self._pid.set_sample_time_us(pid_config['sample_time_us'])
            self._pid.set_output_limits(
                pid_config['output_limits']['min'],
                pid_config['output_limits']['max']
            )
            
            # Set to automatic mode
            self._pid.set_mode(Control.AUTOMATIC)
            
            logger.info("PID controller created and configured")
            
        except Exception as e:
            logger.error(f"Failed to create PID controller: {e}")
            sys.exit(1)
            
    def _should_auto_tune(self) -> bool:
        """Determine if auto-tuning should be performed."""
        auto_tune_config = self.config['auto_tune']
        
        # Check if auto-tuning is enabled
        if not auto_tune_config.get('enabled', False):
            return False
            
        # Auto-tune is enabled
        return True
        
    def _run_auto_tuning(self) -> None:
        """Execute auto-tuning process using STune."""
        logger.info("Starting auto-tuning process")
        
        if not self._input_var or not self._output_var:
            logger.error("Input and output variables must be set for auto-tuning")
            sys.exit(1)
            
        try:
            auto_tune_config = self.config['auto_tune']
            
            # Parse tuning method and action from config
            tuning_method = TuningMethod[auto_tune_config['tuning_method']]
            action = TunerAction[auto_tune_config['action']]
            
            # Create STune instance
            self._s_tune = STune(
                input_var=self._input_var,
                output_var=self._output_var,
                tuning_method=tuning_method,
                action=action,
                serial_mode=SerialMode.PRINT_SUMMARY,
                timer=self.timer
            )
            
            # Configure STune parameters
            self._s_tune.configure(
                input_span=auto_tune_config['input_span'],
                output_span=auto_tune_config['output_span'],
                output_start=auto_tune_config['output_start'],
                output_step=auto_tune_config['output_step'],
                test_time_sec=auto_tune_config['test_time_sec'],
                settle_time_sec=auto_tune_config['settle_time_sec'],
                samples=auto_tune_config['samples']
            )
            
            # Set emergency stop
            self._s_tune.set_emergency_stop(auto_tune_config['emergency_stop'])
            
            logger.info("STune configured, starting auto-tuning test")
            
            # Run auto-tuning with timeout
            start_time = self.timer.get_time_s()
            max_duration = auto_tune_config['max_duration_sec']
            tuning_complete = False
            
            while not tuning_complete:
                current_time = self.timer.get_time_s()
                
                # Check timeout
                if current_time - start_time > max_duration:
                    logger.error(f"Auto-tuning timeout after {max_duration}s")
                    sys.exit(1)
                    
                try:
                    status = self._s_tune.run()
                    
                    if status == TunerStatus.TUNINGS:
                        tuning_complete = True
                        logger.info("Auto-tuning completed successfully")
                        break
                        
                except TuningError as e:
                    logger.error(f"Auto-tuning failed: {e}")
                    sys.exit(1)
                    
                # Small delay to prevent tight loop
                time.sleep(0.01)
                
            # Extract tuned parameters
            kp, ki, kd = self._s_tune.get_auto_tunings()
            
            logger.info(f"Auto-tuned PID parameters: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")
            
            # Update configuration
            self.config['pid']['kp'] = float(kp)
            self.config['pid']['ki'] = float(ki)
            self.config['pid']['kd'] = float(kd)
            
            # Update PID controller with new parameters
            self._pid.set_tunings(kp, ki, kd)
            
            # Save updated configuration
            self._save_config()
            
            self._auto_tuning_complete = True
            logger.info("Auto-tuning process completed and configuration saved")
            
        except Exception as e:
            logger.error(f"Auto-tuning process failed: {e}")
            sys.exit(1)
            
    def _log_to_database(self) -> None:
        """Log current PID state to database."""
        if not self._db_connection:
            return
            
        current_time = self.timer.get_time_s()
        log_interval = self.config['logging']['interval_sec']
        
        # Check if it's time to log
        if current_time - self._last_log_time < log_interval:
            return
            
        try:
            # Get current values
            if not self._input_var or not self._setpoint_var:
                return
                
            pv = self._input_var()
            sp = self._setpoint_var()
            error = sp - pv
            
            # Get PID terms
            p_term = self._pid.get_p_term()
            i_term = self._pid.get_i_term()
            d_term = self._pid.get_d_term()
            output = p_term + i_term + d_term
            
            # Insert into database
            self._db_connection.execute('''
                INSERT INTO pid_logs 
                (timestamp, setpoint, process_variable, control_output, error, p_term, i_term, d_term)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (current_time, sp, pv, output, error, p_term, i_term, d_term))
            
            self._db_connection.commit()
            self._last_log_time = current_time
            
        except Exception as e:
            logger.warning(f"Failed to log to database: {e}")
            
    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._db_connection:
            self._db_connection.close()
            self._db_connection = None
            
    # AdamPID Interface Delegation
    
    def compute(self) -> bool:
        """
        Perform PID calculation and log data.
        
        Returns:
            True if calculation was performed, False if not time yet
        """
        if not self._pid:
            logger.error("PID controller not initialized")
            return False
            
        # Perform PID calculation
        result = self._pid.compute()
        
        # Log to database if calculation was performed
        if result:
            self._log_to_database()
            
        return result
        
    def set_tunings(self, kp: float, ki: float, kd: float, 
                   p_mode: Optional[PMode] = None,
                   d_mode: Optional[DMode] = None,
                   i_aw_mode: Optional[IAwMode] = None) -> None:
        """Set PID tuning parameters and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
            
        self._pid.set_tunings(kp, ki, kd, p_mode, d_mode, i_aw_mode)
        
        # Update configuration
        self.config['pid']['kp'] = float(kp)
        self.config['pid']['ki'] = float(ki)
        self.config['pid']['kd'] = float(kd)
        
        if p_mode is not None:
            self.config['pid']['p_mode'] = p_mode.name
        if d_mode is not None:
            self.config['pid']['d_mode'] = d_mode.name
        if i_aw_mode is not None:
            self.config['pid']['i_aw_mode'] = i_aw_mode.name
            
        # Save configuration
        self._save_config()
        logger.info(f"PID tunings updated: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")
        
    def set_sample_time_us(self, new_sample_time_us: int) -> None:
        """Set sample time and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
            
        self._pid.set_sample_time_us(new_sample_time_us)
        self.config['pid']['sample_time_us'] = new_sample_time_us
        self._save_config()
        
    def set_output_limits(self, min_output: float, max_output: float) -> None:
        """Set output limits and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
            
        self._pid.set_output_limits(min_output, max_output)
        self.config['pid']['output_limits']['min'] = float(min_output)
        self.config['pid']['output_limits']['max'] = float(max_output)
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
        self.config['pid']['action'] = action.name
        self._save_config()
        
    def set_proportional_mode(self, p_mode: PMode) -> None:
        """Set proportional mode and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
            
        self._pid.set_proportional_mode(p_mode)
        self.config['pid']['p_mode'] = p_mode.name
        self._save_config()
        
    def set_derivative_mode(self, d_mode: DMode) -> None:
        """Set derivative mode and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
            
        self._pid.set_derivative_mode(d_mode)
        self.config['pid']['d_mode'] = d_mode.name
        self._save_config()
        
    def set_anti_windup_mode(self, i_aw_mode: IAwMode) -> None:
        """Set anti-windup mode and update configuration."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
            
        self._pid.set_anti_windup_mode(i_aw_mode)
        self.config['pid']['i_aw_mode'] = i_aw_mode.name
        self._save_config()
        
    def set_output_sum(self, sum_value: float) -> None:
        """Set integral sum value directly."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
        self._pid.set_output_sum(sum_value)
        
    def set_input_output_setpoint(self, 
                                 input_var: Callable[[], float],
                                 output_var: Callable[[float], None],
                                 setpoint_var: Callable[[], float]) -> None:
        """Set input, output, and setpoint variable functions."""
        self._input_var = input_var
        self._output_var = output_var
        self._setpoint_var = setpoint_var
        
        if self._pid:
            self._pid.set_input_output_setpoint(input_var, output_var, setpoint_var)
            
    def initialize(self) -> None:
        """Initialize controller for bumpless transfer."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
        self._pid.initialize()
        
    def reset(self) -> None:
        """Reset all internal state variables."""
        if not self._pid:
            logger.error("PID controller not initialized")
            return
        self._pid.reset()
        
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
        
    # Additional AutoAdamPID specific methods
    
    def get_auto_tuning_results(self) -> Dict[str, float]:
        """Get auto-tuning results if available."""
        if not self._s_tune:
            return {}
            
        return {
            'process_gain': self._s_tune.get_process_gain(),
            'dead_time': self._s_tune.get_dead_time(),
            'time_constant': self._s_tune.get_tau(),
            'kp': self._s_tune.get_kp(),
            'ki': self._s_tune.get_ki(),
            'kd': self._s_tune.get_kd()
        }
        
    def is_auto_tuning_complete(self) -> bool:
        """Check if auto-tuning has completed."""
        return self._auto_tuning_complete
        
    def force_config_save(self) -> None:
        """Force save current configuration to file."""
        self._save_config()
        
    def get_database_path(self) -> str:
        """Get path to logging database."""
        return self.config['logging']['database_path']
        
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self._pid:
            return (f"AutoAdamPID(config={self.config_path}, "
                   f"Kp={self.get_kp():.3f}, Ki={self.get_ki():.3f}, "
                   f"Kd={self.get_kd():.3f}, auto_tuned={self._auto_tuning_complete})")
        else:
            return f"AutoAdamPID(config={self.config_path}, not_initialized)"