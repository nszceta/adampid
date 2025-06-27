"""
AdamPID - Professional PID Controller and Autotuner

A Python implementation of advanced PID control and autotuning algorithms,
converted from the QuickPID and sTune C++ libraries.

Copyright (c) 2024 AdamPID Contributors
Licensed under the MIT License.
"""

from .quick_pid import AdamPID, Control, Action, PMode, DMode, IAwMode
from .s_tune import STune, TuningMethod, SerialMode, TunerAction, TunerStatus
from .exceptions import AdamPIDError, TuningError, ConfigurationError

__version__ = "1.0.0"
__author__ = "AdamPID Contributors"
__license__ = "MIT"

__all__ = [
    "AdamPID",
    "STune",
    "Control",
    "Action",
    "PMode",
    "DMode",
    "IAwMode",
    "TuningMethod",
    "SerialMode",
    "AdamPIDError",
    "TuningError",
    "TunerAction",
    "TunerStatus",
    "ConfigurationError",
]
