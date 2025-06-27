"""
Custom exceptions for AdamPID package.
"""


class AdamPIDError(Exception):
    """Base exception for all AdamPID related errors."""
    pass


class TuningError(AdamPIDError):
    """Exception raised when autotuning fails or encounters issues."""
    pass


class ConfigurationError(AdamPIDError):
    """Exception raised when invalid configuration parameters are provided."""
    pass