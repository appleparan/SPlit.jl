"""Optimal train/test split ratio (Joseph 2022)."""

from __future__ import annotations

import numpy as np

from splitiq._convert import DataLike, to_julia_data
from splitiq._julia import _translate_error, julia


def optimal_split_ratio(
    x: DataLike, y: DataLike, *, method: str = 'simple', degree: int = 2
) -> float:
    """Optimal fraction of data to assign to the test set (Joseph 2022, Eq. 11).

    Args:
        x: The predictors: a numpy array-like (1-D or 2-D) or a pandas
            DataFrame.
        y: The response; must have the same number of observations as `x`.
        method: ``'simple'`` counts the encoded predictor columns of `x`
            (after preprocessing: categorical columns Helmert-encoded,
            constant columns dropped) plus one for the intercept.
            ``'regression'`` is the paper's model-selection strategy for an
            unknown model; not implemented in this release.
        degree: Polynomial expansion degree for ``method='regression'``.

    Returns:
        The optimal test-set fraction, ``1 / (sqrt(p) + 1)``.

    Raises:
        ValueError: If `method` is not ``'simple'`` or ``'regression'``, or
            if `x` and `y` have a different number of observations.
        NotImplementedError: If `method` is ``'regression'`` (not yet
            implemented).
    """
    from juliacall import JuliaError

    jl = julia()
    x_julia = to_julia_data(x)
    y_julia = np.asarray(y, dtype=np.float64)
    method_symbol = jl.Symbol(method)
    try:
        with _translate_error():
            value = jl.optimal_split_ratio(x_julia, y_julia, method=method_symbol, degree=degree)
    except JuliaError as exc:
        if jl.isa(exc.exception, jl.ErrorException):
            message = str(jl.sprint(jl.showerror, exc.exception))
            raise NotImplementedError(message) from exc
        raise
    return float(value)
