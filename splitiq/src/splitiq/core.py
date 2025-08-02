"""
Core functions for splitiq package
"""

from typing import Optional, Tuple, Union, Any, Dict
import numpy as np

from .julia_interface import JuliaInterface
from .utils import numpy_to_julia, julia_to_numpy


# Global Julia interface instance
_julia_interface = JuliaInterface()


def split_data(
    X: np.ndarray,
    split_ratio: float = 0.2,
    kappa: Optional[int] = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
    n_threads: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data using optimal support points method.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    split_ratio : float, default=0.2
        Proportion of data to use for test set
    kappa : int, optional
        Subsample size for stochastic optimization. If None, uses full dataset
    max_iterations : int, default=500
        Maximum iterations for optimization
    tolerance : float, default=1e-10
        Convergence tolerance
    n_threads : int, optional
        Number of threads for parallel computation
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    train_indices : np.ndarray
        Indices of training samples
    test_indices : np.ndarray
        Indices of test samples
    """
    from .utils import validate_input_array

    # Validate input
    validate_input_array(X)

    _julia_interface.initialize()

    # Set random seed in Julia if provided
    if random_seed is not None:
        _julia_interface.julia.seval(f"using Random; Random.seed!({random_seed})")

    # Convert to Julia format (ensure it's Float64)
    X_jl = _julia_interface.julia.convert(
        _julia_interface.julia.Matrix[_julia_interface.julia.Float64], X
    )

    # Set up Julia keyword arguments
    kwargs = {}
    kwargs["split_ratio"] = split_ratio
    kwargs["max_iterations"] = max_iterations
    kwargs["tolerance"] = tolerance

    if kappa is not None:
        kwargs["kappa"] = kappa
    if n_threads is not None:
        kwargs["n_threads"] = n_threads

    # Call Julia function
    try:
        # split_data returns just the test indices
        test_indices_jl = _julia_interface.split_jl.split_data(X_jl; **kwargs)

        # Convert to numpy (Julia is 1-based, Python is 0-based)
        test_indices = np.array(test_indices_jl) - 1

        # Generate train indices as complement
        all_indices = np.arange(X.shape[0])
        train_indices = np.setdiff1d(all_indices, test_indices)

        return train_indices.astype(int), test_indices.astype(int)

    except Exception as e:
        from .exceptions import handle_julia_exception
        raise handle_julia_exception(e)


def optimal_split_ratio(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    method: str = "simple",
) -> float:
    """
    Determine optimal split ratio for the dataset.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix
    y : np.ndarray, optional
        Response variable (required for regression method)
    method : str, default="simple"
        Method to use: "simple" or "regression"

    Returns
    -------
    optimal_ratio : float
        Optimal split ratio
    """
    from .utils import validate_input_array

    # Validate input
    validate_input_array(X)

    _julia_interface.initialize()

    # Convert to Julia format
    X_jl = _julia_interface.julia.convert(
        _julia_interface.julia.Matrix[_julia_interface.julia.Float64], X
    )

    try:
        if y is not None:
            y_jl = _julia_interface.julia.convert(
                _julia_interface.julia.Vector[_julia_interface.julia.Float64], y
            )
            ratio = _julia_interface.split_jl.optimal_split_ratio(
                X_jl, y_jl, method=method
            )
        else:
            # For simple method without y, we can compute directly
            ratio = _julia_interface.split_jl.optimal_split_ratio(
                X_jl, X_jl[:, 1], method=method  # Use first column as dummy y
            )

        return float(ratio)

    except Exception as e:
        from .exceptions import handle_julia_exception
        raise handle_julia_exception(e)


def compute_support_points(
    X: np.ndarray,
    n_points: Optional[int] = None,
    kappa: Optional[int] = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """
    Compute support points for the dataset.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix
    n_points : int, optional
        Number of support points. If None, determined automatically
    kappa : int, optional
        Subsample size for stochastic optimization
    max_iterations : int, default=500
        Maximum iterations
    tolerance : float, default=1e-10
        Convergence tolerance
    n_threads : int, optional
        Number of threads for parallel computation

    Returns
    -------
    support_points : np.ndarray
        Computed support points
    """
    from .utils import validate_input_array

    # Validate input
    validate_input_array(X)

    _julia_interface.initialize()

    n_data, p = X.shape

    # Determine number of support points if not specified
    if n_points is None:
        n_points = min(50, max(10, n_data // 10))  # Heuristic

    # Convert to Julia format
    X_jl = _julia_interface.julia.convert(
        _julia_interface.julia.Matrix[_julia_interface.julia.Float64], X
    )

    # Format data (standardize, etc.) using Julia function
    processed_data = _julia_interface.split_jl.format_data(X_jl)

    # Set default kappa
    if kappa is None:
        kappa = n_data

    kwargs = {
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "use_stochastic": kappa < n_data,
    }

    if n_threads is not None:
        kwargs["n_threads"] = n_threads

    try:
        # Call Julia compute_support_points function
        points = _julia_interface.split_jl.compute_support_points(
            n_points, p, processed_data, kappa; **kwargs
        )

        # Convert back to numpy
        return np.array(points)

    except Exception as e:
        from .exceptions import handle_julia_exception
        raise handle_julia_exception(e)
