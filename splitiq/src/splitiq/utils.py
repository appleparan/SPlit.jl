"""
Utility functions for type conversion between Python and Julia
"""

import numpy as np
from typing import Any


def numpy_to_julia(arr: np.ndarray) -> Any:
    """
    Convert NumPy array to Julia array.

    Parameters
    ----------
    arr : np.ndarray
        NumPy array to convert

    Returns
    -------
    julia_array : Any
        Julia array
    """
    # juliacall handles the conversion automatically for most cases
    # but we can add explicit handling if needed
    return arr


def julia_to_numpy(jl_obj: Any) -> np.ndarray:
    """
    Convert Julia object to NumPy array.

    Parameters
    ----------
    jl_obj : Any
        Julia object to convert

    Returns
    -------
    numpy_array : np.ndarray
        NumPy array
    """
    # juliacall usually handles this automatically
    return np.array(jl_obj)


def validate_input_array(X: np.ndarray) -> None:
    """
    Validate input array for SPlit functions.

    Parameters
    ----------
    X : np.ndarray
        Input array to validate

    Raises
    ------
    InputValidationError
        If input is invalid
    """
    from .exceptions import InputValidationError

    if not isinstance(X, np.ndarray):
        raise InputValidationError("Input must be a NumPy array")

    if X.ndim != 2:
        raise InputValidationError("Input must be a 2D array")

    if X.size == 0:
        raise InputValidationError("Input array cannot be empty")

    if np.any(np.isnan(X)):
        raise InputValidationError("Input array contains NaN values")

    if np.any(np.isinf(X)):
        raise InputValidationError("Input array contains infinite values")
