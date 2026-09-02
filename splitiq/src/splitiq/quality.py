"""Split-quality diagnostics: energy distance and squared MMD."""

from __future__ import annotations

from typing import TYPE_CHECKING

from splitiq._convert import DataLike, build_kernel, build_rng, to_julia_data, to_matrix
from splitiq._julia import JuliaValue, _translate_error, julia

if TYPE_CHECKING:
    from splitiq.estimators import DiscrepancyEstimator
    from splitiq.split import SplitResult


def _estimator_kwargs(
    jl: JuliaValue,
    estimator: DiscrepancyEstimator | None,
    seed: int | None,
    n_threads: int | None,
) -> dict[str, JuliaValue]:
    """Build the keyword arguments shared by `energydistance`, `mmd`, `splitquality`.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        estimator: A `DiscrepancyEstimator`, or ``None`` to omit the keyword
            and let Julia pick its own default.
        seed: A seed, or ``None`` to omit the ``rng`` keyword.
        n_threads: Number of threads, or ``None`` to omit the keyword.

    Returns:
        Keyword arguments to splice into the Julia call.
    """
    kwargs: dict[str, JuliaValue] = {}
    if estimator is not None:
        kwargs['estimator'] = estimator._to_julia(jl)
    rng = build_rng(jl, seed)
    if rng is not None:
        kwargs['rng'] = rng
    if n_threads is not None:
        kwargs['n_threads'] = n_threads
    return kwargs


def energydistance(
    x: DataLike,
    y: DataLike,
    *,
    estimator: DiscrepancyEstimator | None = None,
    seed: int | None = None,
    n_threads: int | None = None,
) -> float:
    """Energy distance (Székely & Rizzo) between two samples.

    Args:
        x: A numpy array-like (1-D or 2-D, observations in rows).
        y: A numpy array-like with the same number of columns as `x`.
        estimator: A `DiscrepancyEstimator`, or ``None`` for Julia's default
            (`~splitiq.estimators.Exact`).
        seed: Seed for a fresh RNG; ``None`` uses Julia's default RNG.
        n_threads: Number of threads; ``None`` uses Julia's own default.

    Returns:
        The energy distance between `x` and `y`.

    Raises:
        ValueError: If `x` and `y` have a different number of columns, or
            if `estimator` is not defined for the energy distance.
    """
    jl = julia()
    x_matrix = to_matrix(x)
    y_matrix = to_matrix(y)
    kwargs = _estimator_kwargs(jl, estimator, seed, n_threads)
    with _translate_error():
        return float(jl.energydistance(x_matrix, y_matrix, **kwargs))


def mmd(
    x: DataLike,
    y: DataLike,
    kernel: str = 'gaussian',
    *,
    bandwidth: float | str = 'median',
    estimator: DiscrepancyEstimator | None = None,
    seed: int | None = None,
    n_threads: int | None = None,
) -> float:
    """Squared maximum mean discrepancy (Gretton et al. 2012) between two samples.

    Args:
        x: A numpy array-like (1-D or 2-D, observations in rows).
        y: A numpy array-like with the same number of columns as `x`.
        kernel: ``'energy'`` or ``'gaussian'``.
        bandwidth: A positive number, or ``'median'`` to resolve it from the
            pooled rows of `x` and `y`. Only meaningful for ``'gaussian'``.
        estimator: A `DiscrepancyEstimator`, or ``None`` for Julia's default
            (`~splitiq.estimators.Exact`).
        seed: Seed for a fresh RNG; ``None`` uses Julia's default RNG.
        n_threads: Number of threads; ``None`` uses Julia's own default.

    Returns:
        The squared MMD between `x` and `y` under `kernel`.

    Raises:
        ValueError: If `kernel` is unrecognized, `x` and `y` have a
            different number of columns, or `estimator` is not defined for
            `kernel` (e.g. `~splitiq.estimators.RandomFeatures` with
            ``kernel='energy'``).
    """
    jl = julia()
    x_matrix = to_matrix(x)
    y_matrix = to_matrix(y)
    kernel_obj = build_kernel(jl, kernel, bandwidth)
    kwargs = _estimator_kwargs(jl, estimator, seed, n_threads)
    with _translate_error():
        return float(jl.mmd(x_matrix, y_matrix, kernel_obj, **kwargs))


def splitquality(
    data: DataLike,
    result: SplitResult,
    *,
    kernel: str = 'energy',
    bandwidth: float | str = 'median',
    estimator: DiscrepancyEstimator | None = None,
    exact_threshold: int = 20_000,
    seed: int | None = None,
    n_threads: int | None = None,
) -> float:
    """Discrepancy between the train and test rows of `data`. Smaller is better.

    Args:
        data: The same numpy array-like or pandas DataFrame `result` was
            computed on.
        result: A `SplitResult` from `~splitiq.split.datasplit`.
        kernel: ``'energy'`` or ``'gaussian'``.
        bandwidth: A positive number, or ``'median'`` to resolve it from the
            data. Only meaningful for ``'gaussian'``.
        estimator: A `DiscrepancyEstimator`, or ``None`` to compute exactly
            below `exact_threshold` total rows and fall back to a fixed
            estimator above it (Julia's own default).
        exact_threshold: Row-count threshold below which `estimator=None`
            computes exactly.
        seed: Seed for a fresh RNG; ``None`` uses Julia's default RNG.
        n_threads: Number of threads; ``None`` uses Julia's own default.

    Returns:
        The discrepancy between the train and test rows of `data`.

    Raises:
        ValueError: If `kernel` is unrecognized or Julia rejects the
            arguments.
    """
    jl = julia()
    julia_data = to_julia_data(data)
    kernel_obj = build_kernel(jl, kernel, bandwidth)
    kwargs = _estimator_kwargs(jl, estimator, seed, n_threads)
    kwargs['exact_threshold'] = exact_threshold
    with _translate_error():
        return float(
            jl.splitquality(
                julia_data,
                result._julia_result,
                kernel=kernel_obj,
                **kwargs,
            )
        )
