#!/usr/bin/env python3
"""
Webcam Control with PID Brightness Control
Properly encapsulated design following SOLID principles
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import time

# Assume AdamPID is available as specified
from adampid import AdamPID, Action, Control, SimulatedTimer


class BrightnessMethod(Enum):
    """Available brightness measurement methods"""
    MEAN = "mean"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"  # Best for outliers + speed
    INTERQUARTILE_MEAN = "iq_mean"


@dataclass
class PIDParams:
    """PID parameter configuration"""
    kp: float = 0.100
    ki: float = 0.100
    kd: float = 0.005
    output_min: float = 0.0
    output_max: float = 255.0


@dataclass
class CameraProperty:
    """Camera property information"""
    name: str
    cv_property: int
    value: float
    min_val: float = 0.0
    max_val: float = 255.0


@dataclass
class DiagnosticData:
    """All diagnostic information for display"""
    target_brightness: float
    current_brightness: float
    brightness_error: float
    pid_enabled: bool
    pid_output: float
    p_term: float
    i_term: float
    d_term: float
    kp_gain: float
    ki_gain: float
    kd_gain: float
    control_properties: Dict[str, float]
    fps: float
    frame_count: int


class IBrightnessAnalyzer(ABC):
    """Interface for brightness measurement strategies"""
    
    @abstractmethod
    def measure_brightness(self, frame: np.ndarray) -> float:
        """Measure scene brightness from frame"""
        pass


class BrightnessAnalyzer(IBrightnessAnalyzer):
    """Fast brightness analysis with outlier resistance"""
    
    def __init__(self, method: BrightnessMethod = BrightnessMethod.TRIMMED_MEAN, 
                 downsample_factor: int = 4):
        self.method = method
        self.downsample_factor = downsample_factor
        
        # Pre-compile method function for speed
        self._measurement_func = self._get_measurement_function()
    
    def _get_measurement_function(self) -> Callable[[np.ndarray], float]:
        """Get optimized measurement function based on method"""
        method_map = {
            BrightnessMethod.MEAN: self._measure_mean,
            BrightnessMethod.MEDIAN: self._measure_median,
            BrightnessMethod.TRIMMED_MEAN: self._measure_trimmed_mean,
            BrightnessMethod.INTERQUARTILE_MEAN: self._measure_iq_mean
        }
        return method_map[self.method]
    
    def measure_brightness(self, frame: np.ndarray) -> float:
        """Fast brightness measurement with downsampling"""
        # Fast downsample for speed
        small_frame = frame[::self.downsample_factor, ::self.downsample_factor]
        
        # Convert to grayscale if needed
        if len(small_frame.shape) == 3:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = small_frame
            
        return self._measurement_func(gray)
    
    def _measure_mean(self, gray: np.ndarray) -> float:
        return float(np.mean(gray))
    
    def _measure_median(self, gray: np.ndarray) -> float:
        return float(np.median(gray))
    
    def _measure_trimmed_mean(self, gray: np.ndarray) -> float:
        """Trimmed mean - excellent outlier resistance with good speed"""
        flat = gray.flatten()
        # Remove top and bottom 10%
        trim_size = max(1, len(flat) // 10)
        flat.sort()  # In-place sort for memory efficiency
        trimmed = flat[trim_size:-trim_size] if trim_size < len(flat) // 2 else flat
        return float(np.mean(trimmed))
    
    def _measure_iq_mean(self, gray: np.ndarray) -> float:
        """Interquartile mean - good outlier resistance"""
        q25, q75 = np.percentile(gray, [25, 75])
        mask = (gray >= q25) & (gray <= q75)
        return float(np.mean(gray[mask]))


class CameraManager:
    """Manages camera operations and properties"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.supported_properties: Dict[str, CameraProperty] = {}
        self._initialize_camera()
    
    def _initialize_camera(self) -> None:
        """Initialize camera with best available backend"""
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.camera_id, backend)
                if self.cap and self.cap.isOpened():
                    break
            except Exception:
                continue
        
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set optimal resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self._discover_properties()
    
    def _discover_properties(self) -> None:
        """Discover and catalog supported camera properties"""
        property_definitions = {
            'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
            'CONTRAST': cv2.CAP_PROP_CONTRAST,
            'SATURATION': cv2.CAP_PROP_SATURATION,
            'HUE': cv2.CAP_PROP_HUE,
            'GAIN': cv2.CAP_PROP_GAIN,
            'EXPOSURE': cv2.CAP_PROP_EXPOSURE,
            'GAMMA': cv2.CAP_PROP_GAMMA,
        }
        
        for name, cv_prop in property_definitions.items():
            try:
                value = self.cap.get(cv_prop)
                if value != -1:  # -1 means unsupported
                    self.supported_properties[name] = CameraProperty(
                        name=name,
                        cv_property=cv_prop,
                        value=value
                    )
            except Exception:
                pass
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from camera"""
        if not self.cap:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def set_property(self, property_name: str, value: float) -> bool:
        """Set camera property value"""
        if property_name not in self.supported_properties:
            return False
        
        prop = self.supported_properties[property_name]
        success = self.cap.set(prop.cv_property, value)
        if success:
            prop.value = self.cap.get(prop.cv_property)  # Get actual set value
        return success
    
    def get_property(self, property_name: str) -> Optional[float]:
        """Get current camera property value"""
        if property_name not in self.supported_properties:
            return None
        return self.supported_properties[property_name].value
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution"""
        if not self.cap:
            return 640, 480
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    
    def cleanup(self) -> None:
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None


class BrightnessPIDController:
    """PID controller for automatic brightness control"""
    
    def __init__(self, camera_manager: CameraManager, 
                brightness_analyzer: IBrightnessAnalyzer,
                params: PIDParams = None):
        self.camera_manager = camera_manager
        self.brightness_analyzer = brightness_analyzer
        self.params = params or PIDParams()
        
        # PID state
        self.enabled = False
        self.target_brightness = 128.0  # Default mid-range
        self.current_brightness = 0.0
        self.error = 0.0
        
        # Frame counting for update frequency
        self.update_interval = 2  # Every N frames
        self.frame_count = 0
        
        # AdamPID setup
        self.timer = SimulatedTimer()
        self.pid = AdamPID(
            kp=self.params.kp,
            ki=self.params.ki,
            kd=self.params.kd,
            action=Action.DIRECT,
            timer=self.timer
        )
        
        self.pid.set_output_limits(self.params.output_min, self.params.output_max)
        self.pid.set_mode(Control.MANUAL)  # Start disabled
        self.pid.set_sample_time_us(100_000)  # 100ms
        
        # Initialize setpoint and input to prevent compute() error
        self.pid.set_setpoint(self.target_brightness)
        self.pid.set_input(128.0)  # Initialize with mid-range value
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable PID control"""
        self.enabled = enabled
        if enabled:
            self.pid.set_mode(Control.AUTOMATIC)
            # PID will initialize its own output appropriately
        else:
            self.pid.set_mode(Control.MANUAL)
    
    def set_target_brightness(self, target: float) -> None:
        """Set target brightness (0-255)"""
        self.target_brightness = np.clip(target, 0.0, 255.0)
        self.pid.set_setpoint(self.target_brightness)
    
    def update_pid_params(self, kp: float = None, ki: float = None, kd: float = None) -> None:
        """Update PID parameters"""
        if kp is not None:
            self.params.kp = kp
            self.pid.set_tunings(kp, self.params.ki, self.params.kd)
        if ki is not None:
            self.params.ki = ki
            self.pid.set_tunings(self.params.kp, ki, self.params.kd)
        if kd is not None:
            self.params.kd = kd
            self.pid.set_tunings(self.params.kp, self.params.ki, kd)
    
    def update(self, frame: np.ndarray) -> None:
        """Update PID controller with current frame"""
        self.frame_count += 1
        
        # Only update every N frames for performance
        if self.frame_count % self.update_interval != 0:
            return
        
        # Measure current brightness
        self.current_brightness = self.brightness_analyzer.measure_brightness(frame)
        self.error = self.target_brightness - self.current_brightness
        
        if not self.enabled:
            return
        
        # Advance timer for PID
        self.timer.step(100_000)  # 100ms step
        
        # Update PID
        self.pid.set_input(self.current_brightness)
        
        if self.pid.compute():
            output = self.pid.get_output()
            # Apply to camera
            self.camera_manager.set_property('BRIGHTNESS', output)
    
    def get_diagnostic_data(self) -> Dict[str, float]:
        """Get PID diagnostic information"""
        return {
            'target_brightness': self.target_brightness,
            'current_brightness': self.current_brightness,
            'error': self.error,
            'p_term': self.pid.get_p_term(),
            'i_term': self.pid.get_i_term(),
            'd_term': self.pid.get_d_term(),
            'output': self.pid.get_output(),
            'enabled': self.enabled,
            'kp_gain': self.params.kp,  # Add this line
            'ki_gain': self.params.ki,  # Add this line
            'kd_gain': self.params.kd   # Add this line
        }

class DiagnosticsRenderer:
    """Renders diagnostic information on frames"""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.base_font_scale = 0.6
        self.thickness = 1
        self.color = (0, 255, 0)  # Green
        self.bg_color = (0, 0, 0)  # Black background
    
    def render_diagnostics(self, frame: np.ndarray, data: DiagnosticData) -> np.ndarray:
        """Render all diagnostic information on frame with high fidelity"""
        height, width = frame.shape[:2]
        
        # Much better font scaling for high fidelity
        font_scale = 0.50
        thickness = 1
        
        # Create diagnostic text
        diagnostics = [
            f"PID Control: {'ON' if data.pid_enabled else 'OFF'}",
            f"Target Brightness: {data.target_brightness:.1f}",
            f"Current Brightness: {data.current_brightness:.1f}",
            f"Error: {data.brightness_error:+.1f}",
            f"PID Output: {data.pid_output:.1f}",
            f"P: {data.p_term:.2f} | I: {data.i_term:.2f} | D: {data.d_term:.2f}",
            f"Gains - Kp: {data.kp_gain:.3f} | Ki: {data.ki_gain:.3f} | Kd: {data.kd_gain:.3f}",  # Add this line
            f"FPS: {data.fps:.1f} | Frame: {data.frame_count}",
            "",
            "Camera Properties:",
        ]
        
        # Add camera properties
        for prop_name, value in data.control_properties.items():
            diagnostics.append(f"  {prop_name}: {value:.2f}")
        
        # Add controls
        diagnostics.extend([
            "",
            "Controls:",
            "P/L - PID On/Off  | +/- Target Brightness",
            "1/2/3 - Adjust Kp | 4/5/6 - Adjust Ki | 7/8/9 - Adjust Kd",
            "Q/A - Brightness  | W/S - Contrast | E/D - Saturation",
            "R/F - Hue | T/G - Gamma | Y/H - Gain | U/J - Exposure",
            "SPACE - Reset | ESC - Exit"
        ])
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Better spacing and positioning
        y_offset = int(35 * font_scale)
        line_height = int(35 * font_scale)  # Much more generous spacing
        padding = int(15 * font_scale)
        
        # Calculate total text area for background
        max_width = 0
        total_height = y_offset
        
        for text in diagnostics:
            if text:  # Skip empty lines for width calculation
                (text_width, text_height), _ = cv2.getTextSize(
                    text, self.font, font_scale, thickness
                )
                max_width = max(max_width, text_width)
            total_height += line_height
        
        # Draw semi-transparent background panel
        cv2.rectangle(overlay, 
                    (padding, padding), 
                    (max_width + padding * 3, total_height + padding),
                    self.bg_color, -1)
        
        # Apply transparency (30% background, 70% original)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Reset y_offset for text rendering
        y_offset = int(35 * font_scale) + padding
        
        # Render high-quality text
        for text in diagnostics:
            if text:  # Skip empty lines
                cv2.putText(frame, text, 
                        (padding * 2, y_offset), 
                        self.font, font_scale, self.color, thickness,
                        lineType=cv2.LINE_AA)  # Anti-aliased text for quality
            
            y_offset += line_height
        
        return frame


class InputManager:
    """Handles keyboard input and control logic"""
    
    def __init__(self, camera_manager: CameraManager, 
                 pid_controller: BrightnessPIDController):
        self.camera_manager = camera_manager
        self.pid_controller = pid_controller
        
        # Control mappings
        self.property_controls = {
            'q': ('BRIGHTNESS', 1),   'a': ('BRIGHTNESS', -1),
            'w': ('CONTRAST', 0.05),  's': ('CONTRAST', -0.05),
            'e': ('SATURATION', 0.05), 'd': ('SATURATION', -0.05),
            'r': ('HUE', 1),          'f': ('HUE', -1),
            't': ('GAMMA', 0.05),     'g': ('GAMMA', -0.05),
            'y': ('GAIN', 1),         'h': ('GAIN', -1),
            'u': ('EXPOSURE', 1),     'j': ('EXPOSURE', -1),
        }
        
        self.pid_controls = {
            '1': ('kp', 0.005),   '2': ('kp', -0.005),
            '4': ('ki', 0.005),  '5': ('ki', -0.005),
            '7': ('kd', 0.005),  '8': ('kd', -0.005),
        }
    
    def handle_input(self, key: int) -> bool:
        """Handle keyboard input. Returns True to continue, False to exit"""
        if key == -1:
            return True
        
        char = chr(key) if 0 <= key <= 255 else ''
        
        # Exit
        if key == 27:  # ESC
            return False
        
        # Reset
        elif key == 32:  # SPACE
            self._reset_properties()
        
        # PID control toggle
        elif char == 'p':
            self.pid_controller.set_enabled(True)
        elif char == 'l':
            self.pid_controller.set_enabled(False)
        
        # Target brightness adjustment
        elif char == '=':  # Plus key
            current_target = self.pid_controller.target_brightness
            self.pid_controller.set_target_brightness(current_target + 10)
        elif char == '-':  # Minus key
            current_target = self.pid_controller.target_brightness
            self.pid_controller.set_target_brightness(current_target - 10)
        
        # Camera property controls
        elif char in self.property_controls:
            prop_name, step = self.property_controls[char]
            self._adjust_property(prop_name, step)
        
        # PID parameter controls
        elif char in self.pid_controls:
            param_name, step = self.pid_controls[char]
            self._adjust_pid_param(param_name, step)
        
        # Brightness method cycling (3 key cycles methods)
        elif char == '3':
            self._cycle_brightness_method()
        
        return True
    
    def _adjust_property(self, prop_name: str, step: float) -> None:
        """Adjust camera property by step amount"""
        if prop_name not in self.camera_manager.supported_properties:
            return
        
        current = self.camera_manager.get_property(prop_name)
        if current is not None:
            new_value = current + step
            self.camera_manager.set_property(prop_name, new_value)
    
    def _adjust_pid_param(self, param_name: str, step: float) -> None:
        """Adjust PID parameter by step amount"""
        current_params = {
            'kp': self.pid_controller.params.kp,
            'ki': self.pid_controller.params.ki,
            'kd': self.pid_controller.params.kd
        }
        
        new_value = max(0.0, current_params[param_name] + step)
        
        if param_name == 'kp':
            self.pid_controller.update_pid_params(kp=new_value)
        elif param_name == 'ki':
            self.pid_controller.update_pid_params(ki=new_value)
        elif param_name == 'kd':
            self.pid_controller.update_pid_params(kd=new_value)
    
    def _reset_properties(self) -> None:
        """Reset all properties to defaults"""
        defaults = {
            'BRIGHTNESS': 128,
            'CONTRAST': 1.0,
            'SATURATION': 1.0,
            'HUE': 0.0,
            'GAMMA': 1.0,
        }
        
        for prop_name, default_val in defaults.items():
            self.camera_manager.set_property(prop_name, default_val)
    
    def _cycle_brightness_method(self) -> None:
        """Cycle through brightness measurement methods"""
        # This would require analyzer to support method switching
        # Implementation depends on making BrightnessAnalyzer method switchable
        pass


class WebcamControlApplication:
    """Main application class coordinating all components"""
    
    def __init__(self, camera_id: int = 0):
        # Initialize components following dependency injection
        self.camera_manager = CameraManager(camera_id)
        self.brightness_analyzer = BrightnessAnalyzer()
        self.pid_controller = BrightnessPIDController(
            self.camera_manager, 
            self.brightness_analyzer
        )
        self.diagnostics_renderer = DiagnosticsRenderer()
        self.input_manager = InputManager(self.camera_manager, self.pid_controller)
        
        # Application state
        self.running = False
        self.fps_counter = self._create_fps_counter()
        self.frame_count = 0
    
    def _create_fps_counter(self):
        """Create FPS counter closure"""
        last_time = time.time()
        frame_times = []
        
        def update_fps():
            nonlocal last_time, frame_times
            current_time = time.time()
            frame_times.append(current_time - last_time)
            last_time = current_time
            
            # Keep last 30 frames for averaging
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            return 1.0 / np.mean(frame_times) if frame_times else 0.0
        
        return update_fps
    
    def run(self) -> None:
        """Main application loop"""
        print("WebCam Control with PID Brightness Control")
        print("==========================================")
        print("Controls:")
        print("  P/L - Enable/Disable PID Control")
        print("  +/- - Adjust Target Brightness")
        print("  1/2 - Adjust Kp | 4/5 - Adjust Ki | 7/8 - Adjust Kd")
        print("  ESC - Exit | SPACE - Reset Properties")
        print()
        
        # Create window
        cv2.namedWindow('Webcam PID Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Webcam PID Control', 1200, 800)
        
        self.running = True
        
        try:
            while self.running:
                # Capture frame
                frame = self.camera_manager.get_frame()
                if frame is None:
                    break
                
                self.frame_count += 1
                fps = self.fps_counter()
                
                # Update PID controller
                self.pid_controller.update(frame)
                
                # Gather diagnostic data
                pid_data = self.pid_controller.get_diagnostic_data()
                diagnostic_data = DiagnosticData(
                    target_brightness=pid_data['target_brightness'],
                    current_brightness=pid_data['current_brightness'],
                    brightness_error=pid_data['error'],
                    pid_enabled=pid_data['enabled'],
                    pid_output=pid_data['output'],
                    p_term=pid_data['p_term'],
                    i_term=pid_data['i_term'],
                    d_term=pid_data['d_term'],
                    kp_gain=pid_data['kp_gain'],  # Add this line
                    ki_gain=pid_data['ki_gain'],  # Add this line
                    kd_gain=pid_data['kd_gain'],  # Add this line
                    control_properties={
                        name: prop.value 
                        for name, prop in self.camera_manager.supported_properties.items()
                    },
                    fps=fps,
                    frame_count=self.frame_count
                )
                
                # Render diagnostics
                display_frame = self.diagnostics_renderer.render_diagnostics(
                    frame, diagnostic_data
                )
                
                # Show frame
                cv2.imshow('Webcam PID Control', display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if not self.input_manager.handle_input(key):
                    break
                    
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.running = False
        self.camera_manager.cleanup()
        cv2.destroyAllWindows()


def main():
    """Entry point"""
    try:
        app = WebcamControlApplication()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())