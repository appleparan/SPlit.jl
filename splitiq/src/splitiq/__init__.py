"""
splitiq: Python bindings for SPlit.jl

Optimal data splitting using support points method, powered by Julia backend.
"""

from typing import Optional, Tuple, Union
import numpy as np

# Import Julia interface
from .julia_interface import JuliaInterface
from .core import (
    split_data,
    optimal_split_ratio,
    compute_support_points,
)
from .exceptions import (
    SplitiqError,
    JuliaInitializationError,
    JuliaComputationError,
    InputValidationError,
    ConvergenceError,
)

__version__ = "0.1.0"
__author__ = "Jongsu Kim"
__email__ = "jongsukim8@gmail.com"

# Initialize Julia interface on module import
_julia = JuliaInterface()

# Public API
__all__ = [
    "split_data",
    "optimal_split_ratio",
    "compute_support_points",
    "SplitiqError",
    "JuliaInitializationError",
    "JuliaComputationError",
    "InputValidationError",
    "ConvergenceError",
    "__version__",
]
