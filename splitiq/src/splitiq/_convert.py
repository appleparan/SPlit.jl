"""Conversions between Python/numpy/pandas data and Julia values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from splitiq._julia import JuliaValue

# A numpy array-like (1-D or 2-D) or a pandas DataFrame/Series. Kept dynamic
# since pandas is an optional dependency with no required type stubs here.
type DataLike = Any


def _is_pandas_dataframe(data: DataLike) -> bool:
    """Whether `data` looks like a pandas ``DataFrame``.

    Checked without importing pandas eagerly. A pandas ``Series`` has no
    ``columns`` and is treated as a plain 1-D array-like instead.

    Args:
        data: Candidate object.

    Returns:
        Whether `data` should be treated as a pandas DataFrame.
    """
    if not (hasattr(data, 'columns') and hasattr(data, 'dtypes')):
        return False
    return type(data).__module__.startswith('pandas') or hasattr(data, 'iloc')


def to_julia_data(data: DataLike) -> JuliaValue:
    """Convert a dataset argument (`datasplit`, `splitquality`) to a Julia value.

    Args:
        data: A numpy array-like (1-D or 2-D) or a pandas DataFrame.

    Returns:
        A ``numpy.ndarray`` of ``float64`` (for array-like input, 1-D
        reshaped to a single column) or a Julia ``DataFrame`` built
        column-by-column (for a pandas DataFrame).

    Raises:
        ValueError: If `data` has more than 2 dimensions, contains missing
            values (pandas input), or has a column of an unsupported dtype.
    """
    if _is_pandas_dataframe(data):
        return _dataframe_to_julia(data)
    return to_matrix(data)


def to_matrix(data: DataLike) -> np.ndarray:
    """Convert a numpy array-like sample to a 2-D ``float64`` matrix.

    Args:
        data: A numpy array-like (1-D or 2-D, observations in rows).

    Returns:
        A 2-D ``numpy.ndarray`` of ``float64`` (1-D input reshaped to a
        single column).

    Raises:
        ValueError: If `data` has more than 2 dimensions.
    """
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        msg = f'data must be 1-D or 2-D, got {array.ndim}-D'
        raise ValueError(msg)
    return array


def to_weights(weights: DataLike | None) -> np.ndarray | None:
    """Convert a sample-weights argument to a 1-D ``float64`` vector.

    Args:
        weights: A 1-D array-like with one entry per row, or ``None``.

    Returns:
        A contiguous ``numpy.ndarray`` of ``float64``, or ``None`` when
        `weights` is ``None`` (callers omit the keyword in that case).
        Validation of the values themselves (length, sign, finiteness)
        happens in Julia and surfaces as ``ValueError``.

    Raises:
        ValueError: If `weights` is not one-dimensional.
    """
    if weights is None:
        return None
    array = np.ascontiguousarray(weights, dtype=np.float64)
    if array.ndim != 1:
        msg = f'weights must be 1-D, got {array.ndim}-D'
        raise ValueError(msg)
    return array


def _weights_kwarg(weights: np.ndarray | None) -> dict[str, np.ndarray]:
    """Keyword arguments carrying `weights`, empty when it is ``None``.

    Args:
        weights: A converted weights vector, or ``None``.

    Returns:
        ``{'weights': weights}`` or ``{}``.
    """
    return {} if weights is None else {'weights': weights}


def _dataframe_to_julia(df: DataLike) -> JuliaValue:
    """Build a Julia ``DataFrame`` from a pandas DataFrame, column by column.

    Args:
        df: A pandas DataFrame.

    Returns:
        A Julia ``DataFrame`` with the same column names and row order.
    """
    from splitiq._julia import julia

    jl = julia()
    names: list[str] = []
    julia_columns: list[JuliaValue] = []
    for column_name in df.columns:
        names.append(str(column_name))
        julia_columns.append(_series_to_julia_column(jl, str(column_name), df[column_name]))
    return jl.SplitiqHelpers.build_dataframe(names, julia_columns)


def _series_to_julia_column(jl: JuliaValue, name: str, series: DataLike) -> JuliaValue:
    """Convert one pandas Series to a Julia DataFrame column.

    Numeric (and boolean) dtypes become a ``float64`` vector; ``category``,
    ``object``, and ``string`` dtypes become a Julia ``CategoricalVector``
    (pandas category order is kept; plain string/object columns use the
    sorted unique values as levels, matching SPlit's own canonical order for
    non-categorical columns).

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        name: Column name, used only in the error message.
        series: The pandas Series to convert.

    Returns:
        A Julia value usable as one column of ``SplitiqHelpers.build_dataframe``.

    Raises:
        ValueError: If the column contains missing values or has a dtype
            that is neither numeric, boolean, categorical, nor string/object.
    """
    import pandas as pd

    if series.isna().any():
        msg = 'Dataset contains missing value(s).'
        raise ValueError(msg)

    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = [str(category) for category in series.cat.categories]
        values = series.astype(str).tolist()
        return jl.SplitiqHelpers.categorical_column(values, categories)

    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return np.asarray(series, dtype=np.float64)

    if series.dtype == object or pd.api.types.is_string_dtype(series):
        levels = sorted({str(value) for value in series})
        values = series.astype(str).tolist()
        return jl.SplitiqHelpers.categorical_column(values, levels)

    msg = f'Unsupported dtype for column {name!r}: {series.dtype}'
    raise ValueError(msg)


def to_python_indices(indices: JuliaValue) -> np.ndarray:
    """Convert 1-based Julia row indices to 0-based numpy indices.

    Args:
        indices: A Julia vector of 1-based row indices.

    Returns:
        A 0-based ``numpy.ndarray`` of ``int64``.
    """
    return np.asarray(indices, dtype=np.int64) - 1


def build_rng(jl: JuliaValue, seed: int | None) -> JuliaValue | None:
    """Build a Julia RNG for a `seed`, or nothing for the default RNG.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        seed: A seed, or ``None`` to use Julia's default RNG.

    Returns:
        A Julia ``Random.Xoshiro(seed)`` value, or ``None`` when `seed` is
        ``None`` (callers should omit the ``rng`` keyword in that case).
    """
    if seed is None:
        return None
    return jl.Random.Xoshiro(seed)


def build_kernel(jl: JuliaValue, kernel: str, bandwidth: float | str) -> JuliaValue:
    """Build a Julia ``SplitKernel`` from the Python `kernel`/`bandwidth` args.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        kernel: ``'energy'`` or ``'gaussian'``.
        bandwidth: A positive number, or ``'median'`` (only meaningful for
            ``'gaussian'``; ignored for ``'energy'``).

    Returns:
        A Julia ``EnergyKernel()`` or ``GaussianKernel(...)`` value.

    Raises:
        ValueError: If `kernel` is not ``'energy'`` or ``'gaussian'``.
    """
    if kernel == 'energy':
        return jl.EnergyKernel()
    if kernel == 'gaussian':
        if bandwidth == 'median':
            return jl.GaussianKernel()
        return jl.GaussianKernel(float(bandwidth))
    msg = f"kernel must be 'energy' or 'gaussian', got {kernel!r}"
    raise ValueError(msg)
