"""
Custom exceptions for splitiq package
"""


class SplitiqError(Exception):
    """Base exception for splitiq package"""
    pass


class JuliaInitializationError(SplitiqError):
    """Raised when Julia initialization fails"""
    pass


class JuliaComputationError(SplitiqError):
    """Raised when Julia computation fails"""
    pass


class InputValidationError(SplitiqError):
    """Raised when input validation fails"""
    pass


class ConvergenceError(SplitiqError):
    """Raised when optimization fails to converge"""
    pass


def handle_julia_exception(julia_exception: Exception) -> SplitiqError:
    """
    Convert Julia exceptions to appropriate Python exceptions.

    Parameters
    ----------
    julia_exception : Exception
        The exception raised by Julia

    Returns
    -------
    SplitiqError
        Appropriate Python exception
    """
    error_msg = str(julia_exception)

    # Map common Julia errors to Python exceptions
    if "ArgumentError" in error_msg:
        return InputValidationError(f"Invalid input: {error_msg}")
    elif "BoundsError" in error_msg:
        return InputValidationError(f"Index out of bounds: {error_msg}")
    elif "DimensionMismatch" in error_msg:
        return InputValidationError(f"Dimension mismatch: {error_msg}")
    elif "convergence" in error_msg.lower():
        return ConvergenceError(f"Optimization failed to converge: {error_msg}")
    else:
        return JuliaComputationError(f"Julia computation failed: {error_msg}")
