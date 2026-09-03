"""k-fold multiplets: distribution-balanced folds via SPlit.jl's `multiplet`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from splitiq._convert import (
    DataLike,
    _reference_kwargs,
    _weights_kwarg,
    build_kernel,
    build_rng,
    to_julia_data,
    to_python_indices,
    to_weights,
)
from splitiq._julia import _translate_error, julia
from splitiq.split import SplitKernelName, SplitMethod, StartRule, _build_splitter

if TYPE_CHECKING:
    import numpy as np

MultipletStrategy = Literal['sequential', 'halving', 'single']
_STRATEGIES = ('sequential', 'halving', 'single')


def multiplet(
    data: DataLike,
    k: int,
    *,
    strategy: MultipletStrategy = 'sequential',
    method: SplitMethod = 'twinning',
    kernel: SplitKernelName = 'energy',
    bandwidth: float | Literal['median'] = 'median',
    kappa: int | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
    n_threads: int | None = None,
    seed: int | None = None,
    start: StartRule = 'farthest',
    weights: DataLike | None = None,
    reference: DataLike | None = None,
    reference_weights: DataLike | None = None,
) -> list[np.ndarray]:
    """Partition `data` into `k` folds that each resemble the whole data.

    Mirrors SPlit.jl's ``multiplet`` (Vakayil & Joseph 2022, Section 5).
    The default method is ``'twinning'`` because the paper defines
    multiplets for twinning and ``strategy='single'`` exists only there;
    ``'support_points'`` and ``'herding'`` work with ``'sequential'`` and
    ``'halving'``.

    Args:
        data: A numpy array-like (1-D or 2-D, observations in rows) or a
            pandas DataFrame.
        k: Number of folds, in ``2..len(data)``.
        strategy: ``'sequential'`` (select ``N/k`` rows, then ``N'/(k-1)`` of
            the rest, and so on), ``'halving'`` (split every part in half
            repeatedly; `k` must be a power of two), or ``'single'`` (one
            twinning run, folds by neighbor rank; twinning only).
        method: ``'twinning'`` (default), ``'support_points'``, or
            ``'herding'``.
        kernel: ``'energy'`` or ``'gaussian'`` (twinning: ``'energy'`` only).
        bandwidth: A positive number, or ``'median'`` (Gaussian kernel only).
        kappa: Stochastic subsample size (``'support_points'`` only).
        max_iterations: Maximum optimizer iterations (``'support_points'`` only).
        tolerance: Convergence tolerance (``'support_points'`` only).
        n_threads: Number of threads (not available for twinning).
        seed: Seed for a fresh RNG; ``None`` uses Julia's default RNG.
        start: Twinning's starting row: ``'farthest'``, ``'random'``, or a
            0-based row index.
        weights: One non-negative entry per row, or ``None`` (not available
            for twinning). Cannot be combined with `reference`.
        reference: A dataset of the same kind and columns as `data`, or
            ``None`` (not available for twinning).
        reference_weights: One non-negative entry per row of `reference`,
            or ``None``. Requires `reference`.

    Returns:
        A list of `k` 0-based, ascending index arrays that partition the
        rows; sizes differ by at most one.

    Raises:
        ValueError: If `strategy` or `method` is unrecognized, if an option
            does not apply to `method`, or if Julia rejects the arguments
            (`k` out of range, ``'halving'`` with `k` not a power of two,
            ``'single'`` with a non-twinning method, `weights` with
            twinning, ...).
    """
    if strategy not in _STRATEGIES:
        msg = f'strategy must be one of {_STRATEGIES}, got {strategy!r}'
        raise ValueError(msg)
    if method not in ('support_points', 'herding', 'twinning'):
        msg = f"method must be 'support_points', 'herding', or 'twinning', got {method!r}"
        raise ValueError(msg)
    jl = julia()
    julia_data = to_julia_data(data)
    julia_weights = to_weights(weights)
    julia_reference = to_julia_data(reference) if reference is not None else None
    julia_reference_weights = to_weights(reference_weights)
    kernel_obj = build_kernel(jl, kernel, bandwidth)
    rng = build_rng(jl, seed)
    # `ratio` is never read by `multiplet`.
    splitter = _build_splitter(
        jl, method, kernel, kernel_obj, 0.5, kappa, max_iterations, tolerance, n_threads, rng, start
    )
    with _translate_error():
        folds = jl.multiplet(
            splitter,
            julia_data,
            int(k),
            strategy=jl.Symbol(strategy),
            **_weights_kwarg(julia_weights),
            **_reference_kwargs(julia_reference, julia_reference_weights),
        )
    return [to_python_indices(fold) for fold in folds]
