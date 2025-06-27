"""
Test case demonstrating AdamPID controller and autotuner functionality.

This test simulates a first-order plus dead time (FOPDT) process and demonstrates
both manual PID tuning and automatic tuning using the STune inflection point method.
"""

import time
import math
from adampid import AdamPID, STune, Action, TuningMethod, SerialMode, Control, TunerAction, TunerStatus


class SimpleProcess:
    """
    Simple first-order plus dead time (FOPDT) process simulation.
    
    This simulates a common industrial process model:
    - First-order lag (time constant)
    - Dead time (transport delay)
    - Process gain
    - Noise (optional)
    
    The process equation is:
    y(t) = K * (1 - e^(-(t-td)/tau)) * u(t-td) + noise
    
    Where:
    - K = process gain
    - tau = time constant
    - td = dead time
    - u = input (control signal)
    - y = output (process variable)
    """
    
    def __init__(self, 
                 process_gain: float = 1.0,
                 time_constant: float = 10.0, 
                 dead_time: float = 2.0,
                 noise_level: float = 0.0):
        """
        Initialize the process simulation.
        
        Args:
            process_gain: Steady-state gain between input and output
            time_constant: Time constant (tau) in seconds
            dead_time: Dead time (transport delay) in seconds
            noise_level: Standard deviation of noise to add to output
        """
        self.process_gain = process_gain
        self.time_constant = time_constant
        self.dead_time = dead_time
        self.noise_level = noise_level
        
        # Process state
        self.output = 0.0
        self.steady_state_output = 0.0
        
        # Input history for dead time simulation
        self.input_history = []
        self.time_history = []
        
        # Simulation timing
        self.last_time = time.time()
        self.start_time = self.last_time
    
    def update(self, input_value: float) -> float:
        """
        Update the process with a new input value and return the current output.
        
        Args:
            input_value: Control input to the process
            
        Returns:
            Current process output (process variable)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Store input with timestamp for dead time
        self.input_history.append(input_value)
        self.time_history.append(current_time)
        
        # Remove old history beyond dead time requirement
        while (len(self.time_history) > 1 and 
               current_time - self.time_history[0] > self.dead_time + self.time_constant * 3):
            self.input_history.pop(0)
            self.time_history.pop(0)
        
        # Find input value from dead_time ago
        delayed_input = self._get_delayed_input(current_time - self.dead_time)
        
        # Calculate steady state for current delayed input
        target_output = self.process_gain * delayed_input
        
        # Apply first-order lag: dy/dt = (target - y) / tau
        if dt > 0 and self.time_constant > 0:
            alpha = 1.0 - math.exp(-dt / self.time_constant)
            self.output += alpha * (target_output - self.output)
        
        # Add noise if specified
        if self.noise_level > 0:
            import random
            noise = random.gauss(0, self.noise_level)
            return self.output + noise
        
        return self.output
    
    def _get_delayed_input(self, target_time: float) -> float:
        """Get the input value from a specific time in the past."""
        if not self.time_history:
            return 0.0
        
        # If target time is before our history, return first value
        if target_time <= self.time_history[0]:
            return self.input_history[0]
        
        # If target time is after our latest time, return latest value
        if target_time >= self.time_history[-1]:
            return self.input_history[-1]
        
        # Linear interpolation between two closest points
        for i in range(len(self.time_history) - 1):
            if self.time_history[i] <= target_time <= self.time_history[i + 1]:
                t1, t2 = self.time_history[i], self.time_history[i + 1]
                v1, v2 = self.input_history[i], self.input_history[i + 1]
                
                if t2 - t1 > 0:
                    ratio = (target_time - t1) / (t2 - t1)
                    return v1 + ratio * (v2 - v1)
                else:
                    return v1
        
        return self.input_history[-1]
    
    def reset(self, initial_output: float = 0.0) -> None:
        """Reset the process to initial conditions."""
        self.output = initial_output
        self.input_history.clear()
        self.time_history.clear()
        self.last_time = time.time()
        self.start_time = self.last_time


def test_manual_pid_control():
    """Test manual PID control with known parameters."""
    print("=" * 60)
    print("Testing Manual PID Control")
    print("=" * 60)
    
    # Create process simulation
    process = SimpleProcess(
        process_gain=1.5,      # Output changes 1.5 units per unit input
        time_constant=8.0,     # 8 second time constant  
        dead_time=2.0,         # 2 second dead time
        noise_level=0.01       # Small amount of noise
    )
    
    # Process variables
    setpoint = 50.0
    output = 0.0
    
    # Variable access functions
    def get_input():
        return process.output
    
    def set_output(value):
        nonlocal output
        output = value
    
    def get_setpoint():
        return setpoint
    
    # Create PID controller with manually tuned parameters
    with AdamPID(get_input, set_output, get_setpoint, 
              kp=1.5, ki=0.15, kd=0.4,  # Better gains
              action=Action.DIRECT) as pid:
        
        pid.set_mode(Control.AUTOMATIC)
        pid.set_sample_time_us(100_000)  # 0.1 second sample time
        
        print(f"Process: Gain={process.process_gain}, Tau={process.time_constant}s, Td={process.dead_time}s")
        print(f"PID: Kp={pid.get_kp()}, Ki={pid.get_ki()}, Kd={pid.get_kd()}")
        print(f"Setpoint: {setpoint}")
        print()
        print("Time(s)  Setpoint  Process   Output   Error")
        print("-" * 45)
        
        # Run control loop
        start_time = time.time()
        last_print = start_time
        
        for _ in range(500):  # Run for ~50 seconds
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Update process
            pv = process.update(output)
            
            # Compute PID
            if pid.compute():
                error = setpoint - pv
                
                # Print status every 2 seconds
                if current_time - last_print >= 2.0:
                    print(f"{elapsed:6.1f}   {setpoint:7.1f}   {pv:7.2f}   {output:7.2f}   {error:7.2f}")
                    last_print = current_time
            
            # Stop when reasonably settled
            if elapsed > 20 and abs(setpoint - pv) < 1.0:
                print(f"{elapsed:6.1f}   {setpoint:7.1f}   {pv:7.2f}   {output:7.2f}   {setpoint-pv:7.2f}")
                print(f"\nSettled in {elapsed:.1f} seconds")
                break
            
            time.sleep(0.1)  # 100ms loop time


def test_autotuning():
    """Test automatic PID tuning using STune."""
    print("\n" + "=" * 60)
    print("Testing Automatic PID Tuning with STune")
    print("=" * 60)
    
    # Create a different process for tuning
    process = SimpleProcess(
        process_gain=2.0,      # Different gain
        time_constant=12.0,    # Different time constant
        dead_time=3.0,         # Different dead time
        noise_level=0.02       # Slightly more noise
    )
    
    # Process variables
    setpoint = 0.0  # Start at zero for tuning
    output = 0.0
    
    # Variable access functions
    def get_input():
        return process.output
    
    def set_output(value):
        nonlocal output
        output = value
        
    def get_setpoint():
        return setpoint
    
    print(f"Process: Gain={process.process_gain}, Tau={process.time_constant}s, Td={process.dead_time}s")
    print("Starting autotuning...")
    print()
    
    # Create autotuner
    with STune(get_input, set_output, 
               tuning_method=TuningMethod.ZN_PID,
               action=TunerAction.DIRECT_IP,
               serial_mode=SerialMode.PRINT_SUMMARY) as tuner:
        

        # Reset process to a stable initial state before tuning
        process.reset(0.0)  # Start from zero

        # Configure the autotuning test
        tuner.configure(
            input_span=100.0,
            output_span=100.0,
            output_start=0.0,      # Start from zero
            output_step=25.0,      # 25% step
            test_time_sec=60,
            settle_time_sec=2,     # Reduce settle time
            samples=600
        )
        
        # Run autotuning
        print("Running autotuning test...")
        start_time = time.time()
        last_print = start_time  

        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Update process
            pv = process.update(output)
            
            # Run tuner
            status = tuner.run()
            
            # Print progress every 5 seconds
            if current_time - last_print >= 5.0:
                print(f"  Time: {elapsed:.1f}s, PV: {pv:.2f}, Output: {output:.2f}, Status: {status.name}")
                last_print = current_time
            
            # Check if tuning is complete
            if status == TunerStatus.TUNINGS:
                break
            
            # Safety timeout
            if elapsed > 120:  # 2 minute timeout
                print("Tuning timeout!")
                break
                
            time.sleep(0.1)
        
        # Get tuning results
        kp, ki, kd = tuner.get_auto_tunings()
        print(f"\nAutotuning complete!")
        print(f"Discovered process parameters:")
        print(f"  Process Gain: {tuner.get_process_gain():.3f}")
        print(f"  Dead Time: {tuner.get_dead_time():.3f}s") 
        print(f"  Time Constant: {tuner.get_tau():.3f}s")
        print(f"\nCalculated PID parameters:")
        print(f"  Kp: {kp:.3f}")
        print(f"  Ki: {ki:.3f}") 
        print(f"  Kd: {kd:.3f}")
        
        # Now test the tuned controller
        print(f"\nTesting tuned controller...")
        process.reset(pv)  # Reset process to current state
        setpoint = 75.0    # New setpoint for testing
        
        # Create new PID with tuned parameters
        with AdamPID(get_input, set_output, get_setpoint,
                      kp=kp, ki=ki, kd=kd,
                      action=Action.DIRECT) as pid:
            
            pid.set_mode(Control.AUTOMATIC)
            pid.set_sample_time_us(100_000)  # 0.1 second
            
            print(f"New setpoint: {setpoint}")
            print()
            print("Time(s)  Setpoint  Process   Output   Error")
            print("-" * 45)
            
            start_time = time.time()
            last_print = start_time
            
            for i in range(300):  # Run for ~30 seconds
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Update process
                pv = process.update(output)
                
                # Compute PID
                if pid.compute():
                    error = setpoint - pv
                    
                    # Print every 2 seconds
                    if current_time - last_print >= 2.0:
                        print(f"{elapsed:6.1f}   {setpoint:7.1f}   {pv:7.2f}   {output:7.2f}   {error:7.2f}")
                        last_print = current_time
                
                # Stop when settled
                if elapsed > 15 and abs(setpoint - pv) < 2.0:
                    print(f"{elapsed:6.1f}   {setpoint:7.1f}   {pv:7.2f}   {output:7.2f}   {setpoint-pv:7.2f}")
                    print(f"\nSettled in {elapsed:.1f} seconds with tuned parameters")
                    break
                
                time.sleep(0.1)


if __name__ == "__main__":
    print("AdamPID Test Suite")
    print("=" * 60)
    
    # Get timer information
    from adampid.timing import get_timer_info
    timer_info = get_timer_info()
    print(f"Using timer: {timer_info['description']}")
    print()
    
    # Run tests
    try:
        test_manual_pid_control()
        test_autotuning()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()