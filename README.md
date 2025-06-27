# AdamPID

A professional Python implementation of advanced PID control and autotuning algorithms, converted from the QuickPID and sTune C++ libraries.

## Features

### QuickPID Controller
- Multiple proportional calculation modes (on error, measurement, or both)
- Derivative calculation on error or measurement to reduce derivative kick
- Advanced anti-windup protection with multiple strategies
- Automatic, manual, and timer-based operation modes
- Bumpless transfer between manual and automatic modes
- High-precision timing with automatic fallback

### STune Autotuner
- Inflection point autotuning method (faster than relay tuning)
- Multiple tuning algorithms (Ziegler-Nichols, Cohen-Coon, etc.)
- Works on first-order plus dead time (FOPDT) processes
- Automatic process identification
- 5τ and inflection point test methods

## Installation

### Basic Installation
```bash
pip install adampid

# With Optional Dependencies:

# For plotting capabilities
pip install adampid[plotting]

# For enhanced numerical operations
pip install adampid[enhanced]

# For development
pip install adampid[dev]
```

```bash
# From source

git clone https://github.com/yourusername/adampid.git
cd adampid
pip install -e .
```

### Basic PID Control

```python
from adampid import QuickPID, Action, Control

# Define your process variables
setpoint = 100.0
current_output = 0.0

def get_process_variable():
    # Return current sensor reading
    return read_sensor()

def set_control_output(value):
    # Set actuator output
    global current_output
    current_output = value
    write_actuator(value)

def get_setpoint():
    return setpoint

# Create and configure PID controller
with QuickPID(get_process_variable, set_control_output, get_setpoint,
              kp=2.0, ki=0.1, kd=0.5, action=Action.DIRECT) as pid:
    
    pid.set_mode(Control.AUTOMATIC)
    pid.set_sample_time_us(100_000)  # 100ms sample time
    
    # Main control loop
    while True:
        if pid.compute():
            print(f"PV: {get_process_variable():.2f}, Output: {current_output:.2f}")
        time.sleep(0.01)  # Fast loop, PID handles timing
```

### Automatic Tuning

```python
from adampid import STune, TuningMethod, Action, SerialMode

def get_process_variable():
    return read_sensor()

def set_control_output(value):
    write_actuator(value)

# Create autotuner
with STune(get_process_variable, set_control_output,
           tuning_method=TuningMethod.ZN_PID,
           action=Action.DIRECT_IP,
           serial_mode=SerialMode.PRINT_SUMMARY) as tuner:
    
    # Configure tuning test
    tuner.configure(
        input_span=100.0,      # Full scale input range
        output_span=100.0,     # Full scale output range
        output_start=0.0,      # Starting output
        output_step=25.0,      # Test step size
        test_time_sec=60,      # Max test time
        settle_time_sec=5,     # Settling time
        samples=600            # Number of samples
    )
    
    # Run autotuning
    while True:
        status = tuner.run()
        if status == tuner.TunerStatus.TUNINGS:
            break
        time.sleep(0.1)
    
    # Get results
    kp, ki, kd = tuner.get_auto_tunings()
    print(f"Tuned parameters: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
```

### Multiple PID Calculation Modes

```python
from adampid import QuickPID, PMode, DMode, IAwMode

# Configure advanced PID modes
pid = QuickPID(
    input_var, output_var, setpoint_var,
    kp=1.0, ki=0.1, kd=0.05,
    p_mode=PMode.P_ON_ERROR_MEAS,  # Proportional on error and measurement
    d_mode=DMode.D_ON_MEAS,        # Derivative on measurement (no kick)
    i_aw_mode=IAwMode.I_AW_CONDITION,  # Conditional anti-windup
    action=Action.DIRECT
)

```

### Process Analysis

```python
# After autotuning, analyze process characteristics
process_gain = tuner.get_process_gain()
dead_time = tuner.get_dead_time()
time_constant = tuner.get_tau()

controllability = time_constant / dead_time
if controllability > 0.75:
    print("Process is easy to control")
elif controllability > 0.25:
    print("Process has average controllability")
else:
    print("Process is difficult to control")
```

# API Reference
## QuickPID Class
### Constructor
```python
QuickPID(input_var=None, output_var=None, setpoint_var=None,
         kp=0.0, ki=0.0, kd=0.0, p_mode=PMode.P_ON_ERROR,
         d_mode=DMode.D_ON_MEAS, i_aw_mode=IAwMode.I_AW_CONDITION,
         action=Action.DIRECT)
```

#### Key Methods

`compute()` - Perform PID calculation
`set_tunings(kp, ki, kd)` - Update PID parameters
`set_mode(mode)` - Set operation mode
`set_output_limits(min, max)` - Set output constraints
`set_sample_time_us(microseconds)` - Set timing
`initialize()` - Bumpless transfer setup
`reset()` - Clear all internal states

#### Query Methods

`get_kp()`, `get_ki()`, `get_kd()` - Get tuning parameters
`get_p_term()`, `get_i_term()`, `get_d_term()` - Get individual components
`get_mode()`, `get_direction()` - Get current settings


## STune Class

### Constructor
```python
STune(input_var=None, output_var=None,
      tuning_method=TuningMethod.ZN_PID,
      action=Action.DIRECT_IP,
      serial_mode=SerialMode.PRINT_OFF)
```

#### Key Methods

`configure(input_span, output_span, output_start, output_step, test_time_sec, settle_time_sec, samples)` - Setup test
`run()` - Execute tuning state machine
`reset()` - Reset to initial state
`get_auto_tunings()` - Get calculated PID parameters

#### Process Analysis Methods

`get_process_gain()` - Process steady-state gain
`get_dead_time()` - Process dead time
`get_tau()` - Process time constant

#### Enums

##### Control Modes

`Control.MANUAL` - Manual operation
`Control.AUTOMATIC` - Automatic PID control
`Control.TIMER` - Timer-based calculation
`Control.TOGGLE` - Toggle manual/automatic

##### Controller Actions

`Action.DIRECT` - Direct acting (positive error increases output)
`Action.REVERSE` - Reverse acting (positive error decreases output)

#### Tuning Methods

`TuningMethod.ZN_PID` - Ziegler-Nichols PID
`TuningMethod.DAMPED_OSC_PID` - Damped oscillation PID
`TuningMethod.NO_OVERSHOOT_PID` - No overshoot PID
`TuningMethod.COHEN_COON_PID` - Cohen-Coon PID
`TuningMethod.MIXED_PID` - Mixed method PID
`TuningMethod.*_PI` - Corresponding PI variants

## Best Practices

### Sample Time Selection

Choose sample time ≥ 10x faster than process time constant
For most processes: 100ms to 1000ms sample time
Faster sampling doesn't always improve control

### Autotuning Guidelines

Ensure process is at steady state before tuning
Use appropriate step size (10-25% of output range)
Allow sufficient test time (≥ 3x expected time constant)
Consider process safety during tuning

### PID Tuning Tips

Start with Ziegler-Nichols method
Use damped oscillation for less overshoot
Try no-overshoot method for critical processes
Mixed method provides balanced performance

### Examples
See the test_adampid.py file for complete working examples including:

- Process simulation
- Manual PID tuning
- Automatic tuning workflow
- Performance comparison

## Requirements

- Python 3.8+
- No required dependencies for basic functionality

Optional Dependencies

- matplotlib>=3.5.0 - For plotting capabilities
- numpy>=1.20.0 - For enhanced numerical operations

## License
MIT License - see LICENSE file for details.