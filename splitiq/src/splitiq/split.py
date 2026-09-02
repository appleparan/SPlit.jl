"""Train/test splitting via support points or kernel herding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from splitiq._convert import DataLike, build_kernel, build_rng, to_julia_data, to_python_indices
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
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    converged: bool
    iterations: int
    method: SplitMethod
    kernel: SplitKernelName
    bandwidth: float | None
    ratio: float
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

    Returns:
        The resulting `SplitResult`.

    Raises:
        ValueError: If `method` or `kernel` is unrecognized, if `method` is
            ``'herding'`` and `kappa`/`max_iterations`/`tolerance` are set
            away from their defaults (herding has no such options), or if
            Julia rejects the arguments (e.g. `ratio` outside (0, 1)).
    """
    if method not in ('support_points', 'herding'):
        msg = f"method must be 'support_points' or 'herding', got {method!r}"
        raise ValueError(msg)

    jl = julia()
    julia_data = to_julia_data(data)
    kernel_obj = build_kernel(jl, kernel, bandwidth)
    rng = build_rng(jl, seed)

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
        splitter_kwargs = _splitter_kwargs(kernel_obj, ratio, n_threads, rng)
        with _translate_error():
            splitter = jl.HerdingSplitter(**splitter_kwargs)
            result = jl.datasplit(splitter, julia_data)
    else:
        splitter_kwargs = _splitter_kwargs(kernel_obj, ratio, n_threads, rng)
        splitter_kwargs['kappa'] = kappa
        splitter_kwargs['max_iterations'] = max_iterations
        splitter_kwargs['tolerance'] = tolerance
        with _translate_error():
            splitter = jl.SupportPointSplitter(**splitter_kwargs)
            result = jl.datasplit(splitter, julia_data)

    return _to_split_result(jl, result, method, kernel, ratio)


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
        _julia_result=result,
    )
