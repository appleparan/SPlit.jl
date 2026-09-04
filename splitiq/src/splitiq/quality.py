"""Split-quality diagnostics: energy distance and squared MMD."""

from __future__ import annotations

from typing import TYPE_CHECKING

from splitiq._convert import (
    DataLike,
    _reference_kwargs,
    _weights_kwarg,
    build_kernel,
    build_rng,
    to_julia_data,
    to_matrix,
    to_weights,
)
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
    weights_x: DataLike | None = None,
    weights_y: DataLike | None = None,
) -> float:
    """Energy distance (Székely & Rizzo) between two samples.

    Args:
        x: A numpy array-like (1-D or 2-D, observations in rows).
        y: A numpy array-like with the same number of columns as `x`.
        estimator: A `DiscrepancyEstimator`, or ``None`` for Julia's default
            (`~splitiq.estimators.Exact`).
        seed: Seed for a fresh RNG; ``None`` uses Julia's default RNG.
        n_threads: Number of threads; ``None`` uses Julia's own default.
        weights_x: One non-negative entry per row of `x`, or ``None`` for
            uniform.
        weights_y: Same for `y`. Weights proportional to duplication counts
            are equivalent to duplicating rows.

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
    if weights_x is not None:
        kwargs['weights_x'] = to_weights(weights_x)
    if weights_y is not None:
        kwargs['weights_y'] = to_weights(weights_y)
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
    weights_x: DataLike | None = None,
    weights_y: DataLike | None = None,
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
        weights_x: One non-negative entry per row of `x`, or ``None`` for
            uniform.
        weights_y: Same for `y`. Weights proportional to duplication counts
            are equivalent to duplicating rows.

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
    if weights_x is not None:
        kwargs['weights_x'] = to_weights(weights_x)
    if weights_y is not None:
        kwargs['weights_y'] = to_weights(weights_y)
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
    weights: DataLike | None = None,
    reference: DataLike | None = None,
    reference_weights: DataLike | None = None,
    standardize: bool = True,
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
        weights: One non-negative entry per row of `data`, or ``None``;
            compares the weighted train rows with the weighted test rows.
            Cannot be combined with `reference`.
        reference: A dataset of the same kind and columns as `data`, or
            ``None``. Compares `result`'s selected rows of `data` against
            `reference` instead of against the other side of the split.
        reference_weights: One non-negative entry per row of `reference`,
            or ``None`` for uniform reference weights. Requires `reference`.
        standardize: ``False`` uses a numeric array as it is (no centering,
            scaling, or constant-column removal) — for cosine-normalized
            embeddings; a `~pandas.DataFrame` then raises ``ValueError``.

    Returns:
        The discrepancy between the train and test rows of `data`, or,
        when `reference` is given, between `result`'s selected rows and
        `reference`.

    Raises:
        ValueError: If `kernel` is unrecognized or Julia rejects the
            arguments (e.g. `reference` with a different number of columns
            than `data`, `weights` combined with `reference`, or
            `reference_weights` without `reference`).
    """
    jl = julia()
    julia_data = to_julia_data(data)
    kernel_obj = build_kernel(jl, kernel, bandwidth)
    kwargs = _estimator_kwargs(jl, estimator, seed, n_threads)
    kwargs['exact_threshold'] = exact_threshold
    julia_reference = to_julia_data(reference) if reference is not None else None
    julia_reference_weights = to_weights(reference_weights)
    with _translate_error():
        return float(
            jl.splitquality(
                julia_data,
                result._julia_result,
                kernel=kernel_obj,
                **kwargs,
                **_weights_kwarg(to_weights(weights)),
                **_reference_kwargs(julia_reference, julia_reference_weights),
                standardize=standardize,
            )
        )
