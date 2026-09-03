"""Train/test splitting via support points or kernel herding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

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
from splitiq._julia import JuliaValue, _translate_error, julia

if TYPE_CHECKING:
    from collections.abc import Iterator

_DEFAULT_KAPPA = None
_DEFAULT_MAX_ITERATIONS = 500
_DEFAULT_TOLERANCE = 1e-10

SplitMethod = Literal['support_points', 'herding']
SplitKernelName = Literal['energy', 'gaussian']


@dataclass(frozen=True)
class SplitResult:
    """Outcome of `datasplit`: index partition plus convergence diagnostics.

    Attributes:
        train_indices: 0-based row indices assigned to the train set.
        test_indices: 0-based row indices assigned to the test set.
        converged: Whether the optimizer's stopping rule fired. Always
            ``True`` for kernel herding, which has no iterative convergence
            criterion.
        iterations: Number of optimizer iterations run (kernel herding:
            number of greedy selections).
        method: ``'support_points'`` or ``'herding'``.
        kernel: ``'energy'`` or ``'gaussian'``.
        bandwidth: Resolved Gaussian bandwidth, or ``None`` for the energy
            kernel.
        ratio: Fraction of rows assigned to the test set.
        selected: The side (``'test'`` or ``'train'``) holding the rows the
            splitter chose; the other side is the complement.
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    converged: bool
    iterations: int
    method: SplitMethod
    kernel: SplitKernelName
    bandwidth: float | None
    ratio: float
    selected: Literal['test', 'train']
    _julia_result: JuliaValue = field(repr=False, compare=False)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield `train_indices` then `test_indices`, for tuple unpacking.

        Yields:
            `train_indices`, then `test_indices`.
        """
        yield self.train_indices
        yield self.test_indices

    def apply(self, data: DataLike) -> tuple[DataLike, DataLike]:
        """Split `data` into train and test subsets using this result's indices.

        Args:
            data: A numpy array, or a pandas DataFrame/Series, with the same
                number of rows the split was computed on.

        Returns:
            A ``(train, test)`` tuple of subsets of `data`: fancy indexing
            for a numpy array, ``.iloc`` for a pandas DataFrame or Series.
        """
        if hasattr(data, 'iloc'):
            return data.iloc[self.train_indices], data.iloc[self.test_indices]
        array = np.asarray(data)
        return array[self.train_indices], array[self.test_indices]


def datasplit(
    data: DataLike,
    ratio: float = 0.2,
    *,
    method: SplitMethod = 'support_points',
    kernel: SplitKernelName = 'energy',
    bandwidth: float | Literal['median'] = 'median',
    kappa: int | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
    n_threads: int | None = None,
    seed: int | None = None,
    weights: DataLike | None = None,
    reference: DataLike | None = None,
    reference_weights: DataLike | None = None,
) -> SplitResult:
    """Split `data` into train and test sets whose distributions match closely.

    Args:
        data: A numpy array-like (1-D or 2-D, observations in rows) or a
            pandas DataFrame.
        ratio: Fraction of rows assigned to the test set, in (0, 1).
        method: ``'support_points'`` (Mak & Joseph 2018; Joseph & Vakayil
            2021) or ``'herding'`` (greedy kernel herding).
        kernel: ``'energy'`` or ``'gaussian'``.
        bandwidth: A positive number, or ``'median'`` to resolve it from the
            data. Only meaningful when `kernel` is ``'gaussian'``.
        kappa: Absolute per-iteration subsample size for stochastic
            optimization (``method='support_points'`` only); ``None`` uses
            all rows every iteration.
        max_iterations: Maximum optimizer iterations (``method=
            'support_points'`` only).
        tolerance: Convergence tolerance on the largest squared displacement
            of any support point (``method='support_points'`` only).
        n_threads: Number of threads to use; ``None`` uses Julia's own
            default (``Threads.nthreads()``).
        seed: Seed for a fresh RNG; ``None`` uses Julia's default RNG.
        weights: One non-negative entry per row, or ``None`` for uniform
            weights. Makes the split target the weighted empirical
            distribution of the rows; the selected subset itself is
            uniform. Weights proportional to duplication counts are
            equivalent to duplicating rows. Cannot be combined with
            `reference`.
        reference: A dataset of the same kind and columns as `data`, or
            ``None``. Makes the chosen side approximate the distribution of
            `reference` instead of `data` itself; candidates remain the
            rows of `data`.
        reference_weights: One non-negative entry per row of `reference`,
            or ``None`` for uniform reference weights. Requires `reference`.

    Returns:
        The resulting `SplitResult`.

    Raises:
        ValueError: If `method` or `kernel` is unrecognized, if `method` is
            ``'herding'`` and `kappa`/`max_iterations`/`tolerance` are set
            away from their defaults (herding has no such options), if
            Julia rejects the arguments (e.g. `ratio` outside (0, 1),
            `reference` with a different number of columns than `data`,
            `weights` combined with `reference`, or `reference_weights`
            without `reference`), or if `weights` has the wrong length, a
            negative or non-finite entry, or sums to zero.
    """
    if method not in ('support_points', 'herding'):
        msg = f"method must be 'support_points' or 'herding', got {method!r}"
        raise ValueError(msg)

    jl = julia()
    julia_data = to_julia_data(data)
    julia_weights = to_weights(weights)
    julia_reference = to_julia_data(reference) if reference is not None else None
    julia_reference_weights = to_weights(reference_weights)
    kernel_obj = build_kernel(jl, kernel, bandwidth)
    rng = build_rng(jl, seed)
    splitter = _build_splitter(
        jl, method, kernel_obj, ratio, kappa, max_iterations, tolerance, n_threads, rng
    )
    with _translate_error():
        result = jl.datasplit(
            splitter,
            julia_data,
            **_weights_kwarg(julia_weights),
            **_reference_kwargs(julia_reference, julia_reference_weights),
        )

    return _to_split_result(jl, result, method, kernel, ratio)


def select_rows(
    data: DataLike,
    n: int,
    *,
    method: SplitMethod = 'support_points',
    kernel: SplitKernelName = 'energy',
    bandwidth: float | Literal['median'] = 'median',
    kappa: int | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
    n_threads: int | None = None,
    seed: int | None = None,
    weights: DataLike | None = None,
    reference: DataLike | None = None,
    reference_weights: DataLike | None = None,
) -> np.ndarray:
    """Indices of the `n` rows of `data` the splitter chooses, without a partition.

    The chosen rows approximate the data's own distribution (weighted by
    `weights`) or, when `reference` is given, the distribution of
    `reference` (weighted by `reference_weights`).

    Args:
        data: A numpy array-like (1-D or 2-D, observations in rows) or a
            pandas DataFrame.
        n: Number of rows to select, in ``1:len(data)``.
        method: ``'support_points'`` (Mak & Joseph 2018; Joseph & Vakayil
            2021) or ``'herding'`` (greedy kernel herding).
        kernel: ``'energy'`` or ``'gaussian'``.
        bandwidth: A positive number, or ``'median'`` to resolve it from the
            data. Only meaningful when `kernel` is ``'gaussian'``.
        kappa: Absolute per-iteration subsample size for stochastic
            optimization (``method='support_points'`` only); ``None`` uses
            all rows every iteration.
        max_iterations: Maximum optimizer iterations (``method=
            'support_points'`` only).
        tolerance: Convergence tolerance on the largest squared displacement
            of any support point (``method='support_points'`` only).
        n_threads: Number of threads to use; ``None`` uses Julia's own
            default (``Threads.nthreads()``).
        seed: Seed for a fresh RNG; ``None`` uses Julia's default RNG.
        weights: One non-negative entry per row, or ``None`` for uniform
            weights. Cannot be combined with `reference`.
        reference: A dataset of the same kind and columns as `data`, or
            ``None``. Makes the chosen rows approximate the distribution of
            `reference` instead of `data` itself; candidates remain the
            rows of `data`.
        reference_weights: One non-negative entry per row of `reference`,
            or ``None`` for uniform reference weights. Requires `reference`.

    Returns:
        A 0-based numpy array of `n` row indices, in selection order
        (support-point order for ``method='support_points'``, greedy order
        for ``method='herding'``).

    Raises:
        ValueError: If `method` or `kernel` is unrecognized, if `method` is
            ``'herding'`` and `kappa`/`max_iterations`/`tolerance` are set
            away from their defaults (herding has no such options), or if
            Julia rejects the arguments (e.g. `n` out of range, `reference`
            with a different number of columns than `data`, `weights`
            combined with `reference`, or `reference_weights` without
            `reference`).
    """
    if method not in ('support_points', 'herding'):
        msg = f"method must be 'support_points' or 'herding', got {method!r}"
        raise ValueError(msg)

    jl = julia()
    julia_data = to_julia_data(data)
    julia_weights = to_weights(weights)
    julia_reference = to_julia_data(reference) if reference is not None else None
    julia_reference_weights = to_weights(reference_weights)
    kernel_obj = build_kernel(jl, kernel, bandwidth)
    rng = build_rng(jl, seed)
    # `ratio` only matters for the train/test partition `datasplit` builds;
    # `selectrows` never reads it.
    splitter = _build_splitter(
        jl, method, kernel_obj, 0.5, kappa, max_iterations, tolerance, n_threads, rng
    )
    with _translate_error():
        indices = jl.selectrows(
            splitter,
            julia_data,
            int(n),
            **_weights_kwarg(julia_weights),
            **_reference_kwargs(julia_reference, julia_reference_weights),
        )
    return to_python_indices(indices)


def _build_splitter(
    jl: JuliaValue,
    method: SplitMethod,
    kernel_obj: JuliaValue,
    ratio: float,
    kappa: int | None,
    max_iterations: int,
    tolerance: float,
    n_threads: int | None,
    rng: JuliaValue | None,
) -> JuliaValue:
    """Build the Julia splitter (``SupportPointSplitter``/``HerdingSplitter``) for `method`.

    Assumes `method` is already known to be ``'support_points'`` or
    ``'herding'``.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        method: ``'support_points'`` or ``'herding'``.
        kernel_obj: A Julia ``SplitKernel`` value.
        ratio: Fraction of rows assigned to the test set.
        kappa: Absolute per-iteration subsample size (``method=
            'support_points'`` only).
        max_iterations: Maximum optimizer iterations (``method=
            'support_points'`` only).
        tolerance: Convergence tolerance (``method='support_points'`` only).
        n_threads: Number of threads, or ``None`` to omit the keyword.
        rng: A Julia RNG value, or ``None`` to omit the keyword.

    Returns:
        A Julia ``SupportPointSplitter`` or ``HerdingSplitter`` value.

    Raises:
        ValueError: If `method` is ``'herding'`` and `kappa`/
            `max_iterations`/`tolerance` are set away from their defaults
            (herding has no such options), or if Julia rejects the
            arguments (e.g. `ratio` outside (0, 1)).
    """
    splitter_kwargs = _splitter_kwargs(kernel_obj, ratio, n_threads, rng)
    if method == 'herding':
        herding_options_changed = (
            kappa != _DEFAULT_KAPPA
            or max_iterations != _DEFAULT_MAX_ITERATIONS
            or tolerance != _DEFAULT_TOLERANCE
        )
        if herding_options_changed:
            msg = (
                "herding has no 'kappa'/'max_iterations'/'tolerance' options; "
                'leave them at their defaults'
            )
            raise ValueError(msg)
        with _translate_error():
            return jl.HerdingSplitter(**splitter_kwargs)
    splitter_kwargs['kappa'] = kappa
    splitter_kwargs['max_iterations'] = max_iterations
    splitter_kwargs['tolerance'] = tolerance
    with _translate_error():
        return jl.SupportPointSplitter(**splitter_kwargs)


def _splitter_kwargs(
    kernel_obj: JuliaValue, ratio: float, n_threads: int | None, rng: JuliaValue | None
) -> dict[str, JuliaValue]:
    """Build the keyword arguments shared by both splitter constructors.

    Args:
        kernel_obj: A Julia ``SplitKernel`` value.
        ratio: Fraction of rows assigned to the test set.
        n_threads: Number of threads, or ``None`` to omit the keyword.
        rng: A Julia RNG value, or ``None`` to omit the keyword.

    Returns:
        Keyword arguments for ``SupportPointSplitter``/``HerdingSplitter``.
    """
    kwargs: dict[str, JuliaValue] = {'kernel': kernel_obj, 'ratio': ratio}
    if n_threads is not None:
        kwargs['n_threads'] = n_threads
    if rng is not None:
        kwargs['rng'] = rng
    return kwargs


def _to_split_result(
    jl: JuliaValue, result: JuliaValue, method: SplitMethod, kernel: SplitKernelName, ratio: float
) -> SplitResult:
    """Wrap a Julia ``SplitResult`` in the Python `SplitResult` dataclass.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        result: The Julia ``SplitResult`` returned by ``datasplit``.
        method: The splitting method that produced `result`.
        kernel: The kernel name that produced `result`.
        ratio: The requested test-set fraction.

    Returns:
        The corresponding Python `SplitResult`.
    """
    resolved_bandwidth = float(result.method.kernel.bandwidth) if kernel == 'gaussian' else None
    return SplitResult(
        train_indices=to_python_indices(jl.train_indices(result)),
        test_indices=to_python_indices(jl.test_indices(result)),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        method=method,
        kernel=kernel,
        bandwidth=resolved_bandwidth,
        ratio=ratio,
        selected=_to_selected(result.selected),
        _julia_result=result,
    )


def _to_selected(value: JuliaValue) -> Literal['test', 'train']:
    """Convert the Julia ``SplitResult.selected`` Symbol to a Python literal.

    Args:
        value: The Julia ``Symbol`` (``:test`` or ``:train``) held by
            ``result.selected``.

    Returns:
        ``'test'`` or ``'train'``.

    Raises:
        ValueError: If `value` stringifies to anything else.
    """
    selected = str(value).lstrip(':')
    if selected not in ('test', 'train'):
        msg = f'unexpected SplitResult.selected value from Julia: {selected!r}'
        raise ValueError(msg)
    return selected
