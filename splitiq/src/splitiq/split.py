"""Train/test splitting via support points, kernel herding, or twinning."""

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
_DEFAULT_DELTA = 0.5
_DEFAULT_COMPRESS = 'auto'

SplitMethod = Literal['support_points', 'herding', 'twinning', 'kernel_thinning']
SplitKernelName = Literal['energy', 'gaussian']
StartRule = Literal['farthest', 'random'] | int
CompressMode = Literal['auto', 'always', 'never']
_METHODS = ('support_points', 'herding', 'twinning', 'kernel_thinning')
_COMPRESS_MODES = ('auto', 'always', 'never')


@dataclass(frozen=True)
class SplitResult:
    """Outcome of `datasplit`: index partition plus convergence diagnostics.

    Attributes:
        train_indices: 0-based row indices assigned to the train set.
        test_indices: 0-based row indices assigned to the test set.
        converged: Whether the optimizer's stopping rule fired. Always
            ``True`` for kernel herding and kernel thinning, which have no
            iterative convergence criterion.
        iterations: Number of optimizer iterations run (kernel herding:
            number of greedy selections; kernel thinning: number of
            KT-SWAP replacements).
        method: ``'support_points'``, ``'herding'``, ``'twinning'``, or
            ``'kernel_thinning'``.
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
    start: StartRule | None = None,
    delta: float = 0.5,
    compress: CompressMode = 'auto',
    weights: DataLike | None = None,
    reference: DataLike | None = None,
    reference_weights: DataLike | None = None,
    standardize: bool = True,
) -> SplitResult:
    """Split `data` into train and test sets whose distributions match closely.

    Args:
        data: A numpy array-like (1-D or 2-D, observations in rows) or a
            pandas DataFrame.
        ratio: Fraction of rows assigned to the test set, in (0, 1).
        method: ``'support_points'`` (Mak & Joseph 2018; Joseph & Vakayil
            2022) or ``'herding'`` (greedy kernel herding), ``'twinning'``
            (sequential nearest-neighbor twinning, Vakayil & Joseph 2022;
            energy kernel only, deterministic by default), or
            ``'kernel_thinning'`` (generalized kernel thinning, Dwivedi &
            Mackey 2022/2024; energy or Gaussian kernel; above one half the
            selection is the complement of a kernel-thinning selection of
            the other side).
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
        start: Starting row for ``method='twinning'``: ``'farthest'`` (the
            row farthest from the centroid, deterministic), ``'random'``
            (drawn with `seed`), or a 0-based row index. ``None`` (the
            default) means ``'farthest'`` for ``method='twinning'``; any
            explicit value with another method raises ``ValueError``.
        delta: Failure probability of the kernel-thinning guarantees
            (``method='kernel_thinning'`` only; the papers use ``0.5``).
            Any other value with another method raises ``ValueError``.
        compress: ``'auto'`` (default), ``'always'``, or ``'never'``:
            whether ``method='kernel_thinning'`` runs Compress++ in place of
            plain kernel thinning (``'auto'`` runs it when cheaper at this
            data size; ``'always'`` requires `weights` and `reference` to
            stay unset). A non-default value with another `method` raises
            ``ValueError``.
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
        standardize: ``False`` uses a numeric array as it is (no centering,
            scaling, or constant-column removal) — for cosine-normalized
            embeddings; a `~pandas.DataFrame` then raises ``ValueError``.

    Returns:
        The resulting `SplitResult`.

    Raises:
        ValueError: If `method` or `kernel` is unrecognized, if `method` is
            ``'herding'`` and `kappa`/`max_iterations`/`tolerance` are set
            away from their defaults (herding has no such options), if
            `start` is set for a `method` other than ``'twinning'``, if
            `method` is ``'twinning'`` and `kernel` is not ``'energy'`` or
            `kappa`/`max_iterations`/`tolerance`/`n_threads` are set away
            from their defaults (twinning has no such options), if `delta`
            is set away from its default for a `method` other than
            ``'kernel_thinning'``, if `method` is ``'kernel_thinning'`` and
            `kappa`/`max_iterations`/`tolerance` are set away from their
            defaults (kernel thinning has no such options), if `compress`
            is set away from its default for a `method` other than
            ``'kernel_thinning'``, if `compress` is not one of ``'auto'``,
            ``'always'``, ``'never'``, if Julia
            rejects the arguments (e.g. `ratio`
            outside (0, 1), `delta` outside (0, 1), `reference` with a
            different number of columns than `data`, `weights` combined
            with `reference`, `reference_weights` without `reference`, or
            `compress='always'` combined with `weights` or `reference`),
            or if `weights` has the wrong length, a negative or non-finite
            entry, or sums to zero.
    """
    if method not in _METHODS:
        msg = f'method must be one of {_METHODS}, got {method!r}'
        raise ValueError(msg)

    jl = julia()
    julia_data = to_julia_data(data)
    julia_weights = to_weights(weights)
    julia_reference = to_julia_data(reference) if reference is not None else None
    julia_reference_weights = to_weights(reference_weights)
    kernel_obj = build_kernel(jl, kernel, bandwidth)
    rng = build_rng(jl, seed)
    splitter = _build_splitter(
        jl,
        method,
        kernel,
        kernel_obj,
        ratio,
        kappa,
        max_iterations,
        tolerance,
        n_threads,
        rng,
        start,
        delta,
        compress,
    )
    with _translate_error():
        result = jl.datasplit(
            splitter,
            julia_data,
            **_weights_kwarg(julia_weights),
            **_reference_kwargs(julia_reference, julia_reference_weights),
            standardize=standardize,
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
    start: StartRule | None = None,
    delta: float = 0.5,
    compress: CompressMode = 'auto',
    weights: DataLike | None = None,
    reference: DataLike | None = None,
    reference_weights: DataLike | None = None,
    standardize: bool = True,
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
            2022) or ``'herding'`` (greedy kernel herding), ``'twinning'``
            (sequential nearest-neighbor twinning, Vakayil & Joseph 2022;
            energy kernel only, deterministic by default), or
            ``'kernel_thinning'`` (generalized kernel thinning, Dwivedi &
            Mackey 2022/2024; energy or Gaussian kernel; above one half the
            selection is the complement of a kernel-thinning selection of
            the other side).
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
        start: Starting row for ``method='twinning'``: ``'farthest'`` (the
            row farthest from the centroid, deterministic), ``'random'``
            (drawn with `seed`), or a 0-based row index. ``None`` (the
            default) means ``'farthest'`` for ``method='twinning'``; any
            explicit value with another method raises ``ValueError``.
        delta: Failure probability of the kernel-thinning guarantees
            (``method='kernel_thinning'`` only; the papers use ``0.5``).
            Any other value with another method raises ``ValueError``.
        compress: ``'auto'`` (default), ``'always'``, or ``'never'``:
            whether ``method='kernel_thinning'`` runs Compress++ in place of
            plain kernel thinning (``'auto'`` runs it when cheaper at this
            data size; ``'always'`` requires `weights` and `reference` to
            stay unset). A non-default value with another `method` raises
            ``ValueError``.
        weights: One non-negative entry per row, or ``None`` for uniform
            weights. Cannot be combined with `reference`.
        reference: A dataset of the same kind and columns as `data`, or
            ``None``. Makes the chosen rows approximate the distribution of
            `reference` instead of `data` itself; candidates remain the
            rows of `data`.
        reference_weights: One non-negative entry per row of `reference`,
            or ``None`` for uniform reference weights. Requires `reference`.
        standardize: ``False`` uses a numeric array as it is (no centering,
            scaling, or constant-column removal) — for cosine-normalized
            embeddings; a `~pandas.DataFrame` then raises ``ValueError``.

    Returns:
        A 0-based numpy array of `n` row indices, in selection order
        (support-point order for ``method='support_points'``, greedy order
        for ``method='herding'``, twin-group formation order for
        ``method='twinning'``, coreset position order for
        ``method='kernel_thinning'``).

    Raises:
        ValueError: If `method` or `kernel` is unrecognized, if `method` is
            ``'herding'`` and `kappa`/`max_iterations`/`tolerance` are set
            away from their defaults (herding has no such options), if
            `start` is set for a `method` other than ``'twinning'``, if
            `method` is ``'twinning'`` and `kernel` is not ``'energy'`` or
            `kappa`/`max_iterations`/`tolerance`/`n_threads` are set away
            from their defaults (twinning has no such options), if `delta`
            is set away from its default for a `method` other than
            ``'kernel_thinning'``, if `method` is ``'kernel_thinning'`` and
            `kappa`/`max_iterations`/`tolerance` are set away from their
            defaults (kernel thinning has no such options), if `compress`
            is set away from its default for a `method` other than
            ``'kernel_thinning'``, if `compress` is not one of ``'auto'``,
            ``'always'``, ``'never'``, or if
            Julia rejects the arguments (e.g. `n` out of range, `delta`
            outside (0, 1), `reference` with a different number of columns
            than `data`, `weights` combined with `reference`,
            `reference_weights` without `reference`, or `compress='always'`
            combined with `weights` or `reference`).
    """
    if method not in _METHODS:
        msg = f'method must be one of {_METHODS}, got {method!r}'
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
        jl,
        method,
        kernel,
        kernel_obj,
        0.5,
        kappa,
        max_iterations,
        tolerance,
        n_threads,
        rng,
        start,
        delta,
        compress,
    )
    with _translate_error():
        indices = jl.selectrows(
            splitter,
            julia_data,
            int(n),
            **_weights_kwarg(julia_weights),
            **_reference_kwargs(julia_reference, julia_reference_weights),
            standardize=standardize,
        )
    return to_python_indices(indices)


def _build_splitter(
    jl: JuliaValue,
    method: SplitMethod,
    kernel: SplitKernelName,
    kernel_obj: JuliaValue,
    ratio: float,
    kappa: int | None,
    max_iterations: int,
    tolerance: float,
    n_threads: int | None,
    rng: JuliaValue | None,
    start: StartRule | None,
    delta: float,
    compress: CompressMode,
) -> JuliaValue:
    """Build the Julia splitter for `method`.

    One of ``SupportPointSplitter``, ``HerdingSplitter``, ``TwinningSplitter``,
    or ``KernelThinningSplitter``.

    Assumes `method` is already known to be one of ``_METHODS``.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        method: ``'support_points'``, ``'herding'``, ``'twinning'``, or
            ``'kernel_thinning'``.
        kernel: The kernel name (``'energy'`` or ``'gaussian'``) that built
            `kernel_obj`; used to validate `method='twinning'`, which only
            supports the energy kernel.
        kernel_obj: A Julia ``SplitKernel`` value.
        ratio: Fraction of rows assigned to the test set.
        kappa: Absolute per-iteration subsample size (``method=
            'support_points'`` only).
        max_iterations: Maximum optimizer iterations (``method=
            'support_points'`` only).
        tolerance: Convergence tolerance (``method='support_points'`` only).
        n_threads: Number of threads, or ``None`` to omit the keyword.
        rng: A Julia RNG value, or ``None`` to omit the keyword.
        start: Twinning's starting row (``method='twinning'`` only), or
            ``None`` to use ``'farthest'``; must stay ``None`` for every
            other `method`.
        delta: Kernel thinning's failure probability (``method=
            'kernel_thinning'`` only); must stay at its default for every
            other `method`.
        compress: Kernel thinning's Compress++ mode (``method=
            'kernel_thinning'`` only); must stay at its default (``'auto'``)
            for every other `method`.

    Returns:
        A Julia ``SupportPointSplitter``, ``HerdingSplitter``,
        ``TwinningSplitter``, or ``KernelThinningSplitter`` value.

    Raises:
        ValueError: If `method` is ``'herding'`` and `kappa`/
            `max_iterations`/`tolerance` are set away from their defaults
            (herding has no such options), if `start` is not ``None`` for a
            `method` other than ``'twinning'``, if `method` is
            ``'twinning'`` and any of its unsupported options is set, if
            `delta` is not at its default for a `method` other than
            ``'kernel_thinning'``, if `method` is ``'kernel_thinning'`` and
            `kappa`/`max_iterations`/`tolerance` are set away from their
            defaults, if `compress` is not at its default for a `method`
            other than ``'kernel_thinning'``, if `compress` is not one of
            ``'auto'``, ``'always'``, ``'never'``, or if Julia rejects the
            arguments (e.g. `ratio` outside (0, 1)).
    """
    if start is not None and method != 'twinning':
        msg = "'start' is a twinning option; use method='twinning'"
        raise ValueError(msg)
    if delta != _DEFAULT_DELTA and method != 'kernel_thinning':
        msg = "'delta' is a kernel-thinning option; use method='kernel_thinning'"
        raise ValueError(msg)
    if compress != _DEFAULT_COMPRESS and method != 'kernel_thinning':
        msg = "'compress' is a kernel-thinning option; use method='kernel_thinning'"
        raise ValueError(msg)
    if method == 'twinning':
        return _build_twinning_splitter(
            jl,
            kernel,
            ratio,
            kappa,
            max_iterations,
            tolerance,
            n_threads,
            rng,
            'farthest' if start is None else start,
        )
    if method == 'kernel_thinning':
        return _build_kernel_thinning_splitter(
            jl,
            kernel_obj,
            ratio,
            kappa,
            max_iterations,
            tolerance,
            n_threads,
            rng,
            delta,
            compress,
        )
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


def _build_twinning_splitter(
    jl: JuliaValue,
    kernel: SplitKernelName,
    ratio: float,
    kappa: int | None,
    max_iterations: int,
    tolerance: float,
    n_threads: int | None,
    rng: JuliaValue | None,
    start: StartRule,
) -> JuliaValue:
    """Build a Julia ``TwinningSplitter``; twinning has no kernel, optimizer, or thread options.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        kernel: Must be ``'energy'``.
        ratio: Fraction of rows assigned to the test set.
        kappa: Must be ``None``.
        max_iterations: Must be the default.
        tolerance: Must be the default.
        n_threads: Must be ``None`` (twinning is serial).
        rng: A Julia RNG value, or ``None`` to omit the keyword.
        start: ``'farthest'``, ``'random'``, or a 0-based row index.

    Returns:
        A Julia ``TwinningSplitter`` value.

    Raises:
        ValueError: If any of the constraints above is violated, or if Julia
            rejects the arguments.
    """
    if kernel != 'energy':
        msg = "twinning minimizes the energy distance; use kernel='energy'"
        raise ValueError(msg)
    if (
        kappa != _DEFAULT_KAPPA
        or max_iterations != _DEFAULT_MAX_ITERATIONS
        or tolerance != _DEFAULT_TOLERANCE
    ):
        msg = (
            "twinning has no 'kappa'/'max_iterations'/'tolerance' options; "
            'leave them at their defaults'
        )
        raise ValueError(msg)
    if n_threads is not None:
        msg = "twinning is serial and has no 'n_threads' option"
        raise ValueError(msg)
    kwargs: dict[str, JuliaValue] = {'ratio': ratio, 'start': _to_julia_start(jl, start)}
    if rng is not None:
        kwargs['rng'] = rng
    with _translate_error():
        return jl.TwinningSplitter(**kwargs)


def _build_kernel_thinning_splitter(
    jl: JuliaValue,
    kernel_obj: JuliaValue,
    ratio: float,
    kappa: int | None,
    max_iterations: int,
    tolerance: float,
    n_threads: int | None,
    rng: JuliaValue | None,
    delta: float,
    compress: CompressMode,
) -> JuliaValue:
    """Build a Julia ``KernelThinningSplitter``; it has no optimizer options.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        kernel_obj: A Julia ``SplitKernel`` value (energy or Gaussian).
        ratio: Fraction of rows assigned to the test set.
        kappa: Must be ``None``.
        max_iterations: Must be the default.
        tolerance: Must be the default.
        n_threads: Number of threads, or ``None`` to omit the keyword.
        rng: A Julia RNG value, or ``None`` to omit the keyword.
        delta: Failure probability of the kernel-thinning guarantees, in (0, 1).
        compress: ``'auto'``, ``'always'``, or ``'never'``, forwarded to
            Julia as ``jl.Symbol(compress)``.

    Returns:
        A Julia ``KernelThinningSplitter`` value.

    Raises:
        ValueError: If an optimizer option is set, if `compress` is not one
            of ``'auto'``, ``'always'``, ``'never'``, or if Julia rejects
            the arguments (e.g. `delta` outside (0, 1)).
    """
    if (
        kappa != _DEFAULT_KAPPA
        or max_iterations != _DEFAULT_MAX_ITERATIONS
        or tolerance != _DEFAULT_TOLERANCE
    ):
        msg = (
            "kernel thinning has no 'kappa'/'max_iterations'/'tolerance' options; "
            'leave them at their defaults'
        )
        raise ValueError(msg)
    if compress not in _COMPRESS_MODES:
        msg = f'compress must be one of {_COMPRESS_MODES}, got {compress!r}'
        raise ValueError(msg)
    kwargs = _splitter_kwargs(kernel_obj, ratio, n_threads, rng)
    kwargs['delta'] = float(delta)
    kwargs['compress'] = jl.Symbol(compress)
    with _translate_error():
        return jl.KernelThinningSplitter(**kwargs)


def _to_julia_start(jl: JuliaValue, start: StartRule) -> JuliaValue:
    """Convert `start` to the Julia keyword value (Symbol, or 1-based row index).

    Args:
        jl: The Julia ``Main`` handle.
        start: ``'farthest'``, ``'random'``, or a 0-based row index.

    Returns:
        ``:farthest``/``:random`` or ``start + 1``.

    Raises:
        ValueError: If `start` is a negative int, not an integer, or an unknown string.
    """
    if isinstance(start, str):
        if start not in ('farthest', 'random'):
            msg = f"start must be 'farthest', 'random', or a row index, got {start!r}"
            raise ValueError(msg)
        return jl.Symbol(start)
    if not isinstance(start, (int, np.integer)) or isinstance(start, bool):
        msg = f"start must be 'farthest', 'random', or an integer row index, got {start!r}"
        raise ValueError(msg)
    if int(start) < 0:
        msg = f'start must be a non-negative row index, got {start}'
        raise ValueError(msg)
    return int(start) + 1


def _splitter_kwargs(
    kernel_obj: JuliaValue, ratio: float, n_threads: int | None, rng: JuliaValue | None
) -> dict[str, JuliaValue]:
    """Build the keyword arguments shared by every kernel-based splitter constructor.

    Args:
        kernel_obj: A Julia ``SplitKernel`` value.
        ratio: Fraction of rows assigned to the test set.
        n_threads: Number of threads, or ``None`` to omit the keyword.
        rng: A Julia RNG value, or ``None`` to omit the keyword.

    Returns:
        Keyword arguments for ``SupportPointSplitter``, ``HerdingSplitter``,
        or ``KernelThinningSplitter``.
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
