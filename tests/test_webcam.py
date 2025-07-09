#!/usr/bin/env python3
"""
Webcam Control with PID Brightness Control
Automatic scene brightness control using camera brightness adjustment
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import time

# Assume AdamPID is available as specified
from adampid import AdamPID, IAwMode, Control, RealTimeTimer


class BrightnessMethod(Enum):
    """Available scene brightness measurement methods"""

    MEAN = "mean"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"  # Best for outliers + speed
    INTERQUARTILE_MEAN = "iq_mean"


@dataclass
class PIDParams:
    """PID parameter configuration"""

    kp: float = 0.250
    ki: float = 0.250
    kd: float = 0.020
    output_min: float = -100.0
    output_max: float = 100.0


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

    target_scene_brightness: float
    measured_scene_brightness: float
    brightness_error: float
    pid_enabled: bool
    camera_brightness_output: float
    p_term: float
    i_term: float
    d_term: float
    kp_gain: float
    ki_gain: float
    kd_gain: float
    camera_brightness_setting: float
    fps: float
    frame_count: int


class IBrightnessAnalyzer(ABC):
    """Interface for scene brightness measurement strategies"""

    @abstractmethod
    def measure_scene_brightness(self, frame: np.ndarray) -> float:
        """Measure scene brightness from frame"""
        pass


class BrightnessAnalyzer(IBrightnessAnalyzer):
    """Fast scene brightness analysis with outlier resistance"""

    def __init__(
        self,
        method: BrightnessMethod = BrightnessMethod.TRIMMED_MEAN,
        downsample_factor: int = 4,
    ):
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
            BrightnessMethod.INTERQUARTILE_MEAN: self._measure_iq_mean,
        }
        return method_map[self.method]

    def measure_scene_brightness(self, frame: np.ndarray) -> float:
        """Fast scene brightness measurement with downsampling"""
        # Fast downsample for speed
        small_frame = frame[:: self.downsample_factor, :: self.downsample_factor]

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
    """Manages camera operations and brightness property"""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.brightness_property: Optional[CameraProperty] = None
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

        self._discover_brightness_property()

    def _discover_brightness_property(self) -> None:
        """Discover and check camera brightness property support"""
        try:
            value = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            if value != -1:  # -1 means unsupported
                self.brightness_property = CameraProperty(
                    name="BRIGHTNESS", cv_property=cv2.CAP_PROP_BRIGHTNESS, value=value
                )
            else:
                raise RuntimeError("Camera does not support brightness control")
        except Exception as e:
            raise RuntimeError(f"Could not access camera brightness property: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from camera"""
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        return frame if ret else None

    def set_camera_brightness(self, value: float) -> bool:
        """Set camera brightness value (0-255)"""
        if not self.brightness_property:
            return False

        success = self.cap.set(self.brightness_property.cv_property, value)
        if success:
            self.brightness_property.value = self.cap.get(
                self.brightness_property.cv_property
            )
        return success

    def get_camera_brightness(self) -> float:
        """Get current camera brightness setting"""
        assert self.brightness_property
        return self.brightness_property.value

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


class IncrementalBrightnessPIDController:
    """PID controller for automatic scene brightness control via camera brightness adjustment"""

    def __init__(
        self,
        camera_manager: CameraManager,
        brightness_analyzer: IBrightnessAnalyzer,
        params: PIDParams = PIDParams(),
    ):
        self.camera_manager = camera_manager
        self.brightness_analyzer = brightness_analyzer
        self.params = params

        # PID state
        self.enabled = False
        self.target_scene_brightness = 128.0  # Default mid-range target
        self.measured_scene_brightness = 0.0
        self.error = 0.0

        # Frame counting for update frequency
        self.update_interval = 2  # Every N frames
        self.frame_count = 0

        # AdamPID setup
        self.timer = RealTimeTimer()
        self.pid = AdamPID(
            kp=self.params.kp,
            ki=self.params.ki,
            kd=self.params.kd,
            timer=self.timer,
            i_aw_mode=IAwMode.I_AW_CONDITION,
        )

        # PID output will be CHANGE in brightness, not absolute value
        self.pid.set_output_limits(self.params.output_min, self.params.output_max)
        self.pid.set_mode(Control.MANUAL)  # Start disabled
        self.pid.set_sample_time_us(100_000)  # 100ms

        # Initialize setpoint and input to prevent compute() error
        self.pid.set_setpoint(self.target_scene_brightness)
        self.pid.set_input(0.0)

        # Store current camera brightness as baseline
        self.baseline_camera_brightness = (
            camera_manager.get_camera_brightness() or 128.0
        )

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable PID control"""
        self.enabled = enabled
        if enabled:
            self.pid.set_mode(Control.AUTOMATIC)
        else:
            self.pid.set_mode(Control.MANUAL)

    def set_target_scene_brightness(self, target: float) -> None:
        """Set target scene brightness (0-255)"""
        self.target_scene_brightness = np.clip(target, 0.0, 255.0)
        self.pid.set_setpoint(self.target_scene_brightness)

    def update_pid_params(
        self, kp: float = None, ki: float = None, kd: float = None
    ) -> None:
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

        # Measure current scene brightness
        self.measured_scene_brightness = (
            self.brightness_analyzer.measure_scene_brightness(frame)
        )
        self.error = self.target_scene_brightness - self.measured_scene_brightness

        if not self.enabled:
            return

        # Update PID
        self.pid.set_input(self.measured_scene_brightness)

        if self.enabled and self.pid.compute():
            # PID output is now a CHANGE in camera brightness
            brightness_change = self.pid.get_output()

            # Apply change to current camera brightness
            current_brightness = self.camera_manager.get_camera_brightness()
            new_brightness = np.clip(current_brightness + brightness_change, 0, 255)

            self.camera_manager.set_camera_brightness(new_brightness)

    def get_diagnostic_data(self) -> Dict[str, float]:
        """Get PID diagnostic information"""
        return {
            "target_scene_brightness": self.target_scene_brightness,
            "measured_scene_brightness": self.measured_scene_brightness,
            "error": self.error,
            "p_term": self.pid.get_p_term(),
            "i_term": self.pid.get_i_term(),
            "d_term": self.pid.get_d_term(),
            "camera_brightness_output": self.pid.get_output(),
            "enabled": self.enabled,
            "kp_gain": self.params.kp,
            "ki_gain": self.params.ki,
            "kd_gain": self.params.kd,
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
        """Render all diagnostic information on frame"""
        height, width = frame.shape[:2]

        font_scale = 0.50
        thickness = 1

        # Create diagnostic text
        diagnostics = [
            f"PID Control: {'ON' if data.pid_enabled else 'OFF'}",
            f"Target Scene Brightness: {data.target_scene_brightness:.1f}",
            f"Measured Scene Brightness: {data.measured_scene_brightness:.1f}",
            f"Error: {data.brightness_error:+.1f}",
            f"Camera Brightness Output: {data.camera_brightness_output:.1f}",
            f"Camera Brightness Setting: {data.camera_brightness_setting:.1f}",
            f"P: {data.p_term:.2f} | I: {data.i_term:.2f} | D: {data.d_term:.2f}",
            f"Gains - Kp: {data.kp_gain:.3f} | Ki: {data.ki_gain:.3f} | Kd: {data.kd_gain:.3f}",
            f"FPS: {data.fps:.1f} | Frame: {data.frame_count}",
            "",
            "Controls:",
            "P/L - PID On/Off  | +/- Target Scene Brightness",
            "1/2 - Adjust Kp | 4/5 - Adjust Ki | 7/8 - Adjust Kd",
            "ESC - Exit",
        ]

        # Create overlay for transparency
        overlay = frame.copy()

        y_offset = int(35 * font_scale)
        line_height = int(35 * font_scale)
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
        cv2.rectangle(
            overlay,
            (padding, padding),
            (max_width + padding * 3, total_height + padding),
            self.bg_color,
            -1,
        )

        # Apply transparency
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Reset y_offset for text rendering
        y_offset = int(35 * font_scale) + padding

        # Render text
        for text in diagnostics:
            if text:  # Skip empty lines
                cv2.putText(
                    frame,
                    text,
                    (padding * 2, y_offset),
                    self.font,
                    font_scale,
                    self.color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

            y_offset += line_height

        return frame


class InputManager:
    """Handles keyboard input for PID brightness control"""

    def __init__(
        self,
        camera_manager: CameraManager,
        pid_controller: IncrementalBrightnessPIDController,
    ):
        self.camera_manager = camera_manager
        self.pid_controller = pid_controller

        # PID parameter controls only
        self.pid_controls = {
            "1": ("kp", -0.005),
            "2": ("kp", 0.005),
            "4": ("ki", -0.005),
            "5": ("ki", 0.005),
            "7": ("kd", -0.005),
            "8": ("kd", 0.005),
        }

    def handle_input(self, key: int) -> bool:
        """Handle keyboard input. Returns True to continue, False to exit"""
        if key == -1:
            return True

        char = chr(key) if 0 <= key <= 255 else ""

        # Exit
        if key == 27:  # ESC
            return False

        # PID control toggle
        elif char == "p":
            self.pid_controller.set_enabled(True)
        elif char == "l":
            self.pid_controller.set_enabled(False)

        # Target scene brightness adjustment
        elif char == "=":  # Plus key
            current_target = self.pid_controller.target_scene_brightness
            self.pid_controller.set_target_scene_brightness(current_target + 10)
        elif char == "-":  # Minus key
            current_target = self.pid_controller.target_scene_brightness
            self.pid_controller.set_target_scene_brightness(current_target - 10)

        # PID parameter controls
        elif char in self.pid_controls:
            param_name, step = self.pid_controls[char]
            self._adjust_pid_param(param_name, step)

        return True

    def _adjust_pid_param(self, param_name: str, step: float) -> None:
        """Adjust PID parameter by step amount"""
        current_params = {
            "kp": self.pid_controller.params.kp,
            "ki": self.pid_controller.params.ki,
            "kd": self.pid_controller.params.kd,
        }

        new_value = max(0.0, current_params[param_name] + step)

        if param_name == "kp":
            self.pid_controller.update_pid_params(kp=new_value)
        elif param_name == "ki":
            self.pid_controller.update_pid_params(ki=new_value)
        elif param_name == "kd":
            self.pid_controller.update_pid_params(kd=new_value)


class WebcamControlApplication:
    """Main application class for PID brightness control"""

    def __init__(self, camera_id: int = 0):
        # Initialize components following dependency injection
        self.camera_manager = CameraManager(camera_id)
        self.brightness_analyzer = BrightnessAnalyzer()
        self.pid_controller = IncrementalBrightnessPIDController(
            self.camera_manager, self.brightness_analyzer
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
        print("WebCam PID Brightness Control")
        print("=============================")
        print("Automatic scene brightness control via camera brightness adjustment")
        print("")
        print("Controls:")
        print("  P/L - Enable/Disable PID Control")
        print("  +/- - Adjust Target Scene Brightness")
        print("  1/2 - Adjust Kp | 4/5 - Adjust Ki | 7/8 - Adjust Kd")
        print("  ESC - Exit")
        print()

        # Create window
        cv2.namedWindow("Webcam PID Brightness Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam PID Brightness Control", 1200, 800)

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
                camera_brightness = self.camera_manager.get_camera_brightness() or 0.0

                diagnostic_data = DiagnosticData(
                    target_scene_brightness=pid_data["target_scene_brightness"],
                    measured_scene_brightness=pid_data["measured_scene_brightness"],
                    brightness_error=pid_data["error"],
                    pid_enabled=pid_data["enabled"],
                    camera_brightness_output=pid_data["camera_brightness_output"],
                    p_term=pid_data["p_term"],
                    i_term=pid_data["i_term"],
                    d_term=pid_data["d_term"],
                    kp_gain=pid_data["kp_gain"],
                    ki_gain=pid_data["ki_gain"],
                    kd_gain=pid_data["kd_gain"],
                    camera_brightness_setting=camera_brightness,
                    fps=fps,
                    frame_count=self.frame_count,
                )

                # Render diagnostics
                display_frame = self.diagnostics_renderer.render_diagnostics(
                    frame, diagnostic_data
                )

                # Show frame
                cv2.imshow("Webcam PID Brightness Control", display_frame)

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
