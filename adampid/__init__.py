"""
AdamPID - Professional PID Controller and Autotuner

A Python implementation of advanced PID control and autotuning algorithms,
converted from the QuickPID and sTune C++ libraries.

Copyright (c) 2024 AdamPID Contributors
Licensed under the MIT License.
"""

from .adampid import Action, AdamPID, Control, DMode, IAwMode, PMode
from .auto_adampid import AutoAdamPID, AutoAdamPIDError
from .exceptions import AdamPIDError, ConfigurationError, TuningError
from .real_time_timer import RealTimeTimer
from .s_tune import SerialMode, STune, TunerAction, TunerStatus, TuningMethod
from .simulated_timer import SimulatedTimer
from .timing_base import TimerBase

__version__ = "1.0.0"
__author__ = "AdamPID Contributors"
__license__ = "MIT"

__all__ = [
    "AdamPID",
    "AutoAdamPID",
    "STune",
    "Control",
    "Action",
    "PMode",
    "DMode",
    "IAwMode",
    "TuningMethod",
    "SerialMode",
    "AdamPIDError",
    "AutoAdamPIDError",
    "TuningError",
    "TunerAction",
    "TunerStatus",
    "ConfigurationError",
    "TimerBase",
    "RealTimeTimer",
    "SimulatedTimer",
]
