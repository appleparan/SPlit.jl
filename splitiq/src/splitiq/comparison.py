"""Side-by-side comparison of splitter configurations on one dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from splitiq._convert import (
    DataLike,
    _reference_kwargs,
    _weights_kwarg,
    build_kernel,
    build_rng,
    to_julia_data,
    to_weights,
)
from splitiq._julia import JuliaValue, _translate_error, julia
from splitiq.quality import _estimator_kwargs
from splitiq.split import (
    _DEFAULT_COMPRESS,
    _DEFAULT_DELTA,
    _DEFAULT_KAPPA,
    _DEFAULT_MAX_ITERATIONS,
    _DEFAULT_TOLERANCE,
    _METHODS,
    CompressMode,
    SplitKernelName,
    SplitMethod,
    SplitResult,
    StartRule,
    _build_splitter,
    _to_split_result,
)

if TYPE_CHECKING:
    from splitiq.estimators import DiscrepancyEstimator

_METHOD_OPTION_KEYS = (
    'kernel',
    'bandwidth',
    'kappa',
    'max_iterations',
    'tolerance',
    'start',
    'delta',
    'compress',
)


@dataclass(frozen=True)
class _MethodSpec:
    """One normalized entry of `compare`'s `methods`, defaults filled in."""

    method: SplitMethod
    kernel: SplitKernelName
    bandwidth: float | Literal['median']
    kappa: int | None
    max_iterations: int
    tolerance: float
    start: StartRule | None
    delta: float
    compress: CompressMode


@dataclass(frozen=True)
class SplitComparison:
    """Outcome of `compare`: one `SplitResult` and one quality per method spec.

    `results` and `qualities` are index-aligned with the `methods` sequence
    passed to `compare`.

    Attributes:
        results: One `SplitResult` per entry of `methods`, in the same order.
        qualities: One discrepancy score per entry of `results`, under
            `kernel`; lower is better.
        kernel: The scoring kernel every quality was computed under
            (``'energy'`` or ``'gaussian'``) — not each splitter's own
            kernel, which may differ per method spec.
    """

    results: tuple[SplitResult, ...]
    qualities: tuple[float, ...]
    kernel: SplitKernelName

    def best(self) -> tuple[int, SplitResult]:
        """The index and result with the lowest discrepancy.

        Returns:
            ``(index, result)`` for the entry of `qualities` that is
            smallest.
        """
        index = min(range(len(self.qualities)), key=self.qualities.__getitem__)
        return index, self.results[index]


def compare(
    data: DataLike,
    methods: Sequence[SplitMethod | Mapping[str, object]],
    *,
    ratio: float = 0.2,
    kernel: SplitKernelName = 'energy',
    bandwidth: float | Literal['median'] = 'median',
    estimator: DiscrepancyEstimator | None = None,
    exact_threshold: int = 20_000,
    seed: int | None = None,
    n_threads: int | None = None,
    weights: DataLike | None = None,
    reference: DataLike | None = None,
    reference_weights: DataLike | None = None,
    standardize: bool = True,
) -> SplitComparison:
    """Run `datasplit` with each of `methods` on `data` and score every split.

    Every split is scored with `splitquality` under `kernel` (the *scoring*
    kernel), which may differ from any per-method `kernel` used to build a
    splitter itself.

    Args:
        data: A numpy array-like (1-D or 2-D, observations in rows) or a
            pandas DataFrame.
        methods: A sequence of method names (``'support_points'``,
            ``'herding'``, ``'twinning'``, ``'kernel_thinning'``, built with
            default options) or mappings with a required ``'method'`` key
            and any of the per-method keywords `~splitiq.split.datasplit`
            accepts (``kernel``, ``bandwidth``, ``kappa``,
            ``max_iterations``, ``tolerance``, ``start``, ``delta``,
            ``compress``).
        ratio: Fraction of rows assigned to the test set, shared by every
            splitter.
        kernel: ``'energy'`` or ``'gaussian'``: the discrepancy every split
            is scored under, passed to `splitquality`. Independent of any
            per-method ``kernel`` in `methods`.
        bandwidth: A positive number, or ``'median'`` to resolve it from the
            data. Only meaningful when `kernel` is ``'gaussian'``.
        estimator: A `DiscrepancyEstimator` for the scoring step, or
            ``None`` to compute exactly below `exact_threshold` total rows
            and fall back to a fixed estimator above it (Julia's own
            default). Passed to `~splitiq.quality.splitquality`; it does
            not change the splitters themselves.
        exact_threshold: Row-count threshold below which `estimator=None`
            scores exactly. Passed to `~splitiq.quality.splitquality`.
        seed: Seed shared by every splitter and by the scoring kernel's
            ``'median'`` bandwidth resolution; each splitter gets its own
            fresh RNG built from this seed (as if each were constructed
            with its own ``Random.Xoshiro(seed)``), so they see the same
            random stream. ``None`` uses Julia's default RNG throughout.
        n_threads: Number of threads to use; ``None`` uses Julia's own
            default (``Threads.nthreads()``).
        weights: One non-negative entry per row, or ``None`` for uniform
            weights. Forwarded to every splitter's `datasplit` call and to
            `splitquality`. Cannot be combined with `reference`.
        reference: A dataset of the same kind and columns as `data`, or
            ``None``. Forwarded to every splitter's `datasplit` call and to
            `splitquality`, and the scoring kernel's ``'median'`` bandwidth
            is then resolved on `reference` instead of `data`.
        reference_weights: One non-negative entry per row of `reference`,
            or ``None`` for uniform reference weights. Requires `reference`.
        standardize: ``False`` uses a numeric array as it is (no centering,
            scaling, or constant-column removal) — for cosine-normalized
            embeddings; a `~pandas.DataFrame` then raises ``ValueError``.

    Returns:
        A `SplitComparison` with one result and one quality per entry of
        `methods`, index-aligned.

    Raises:
        ValueError: If `methods` is empty, if an entry of `methods` is not a
            recognized method name or a mapping with a ``'method'`` key
            naming one, if a mapping has a key outside the per-method
            option set above, if any per-method option is invalid for its
            `method` (see `~splitiq.split.datasplit`), if `estimator` is
            not defined for `kernel`, or if Julia rejects the arguments
            (e.g. `ratio` outside (0, 1), `weights` combined with
            `reference`, `reference_weights` without `reference`).
    """
    if not methods:
        msg = '`methods` must not be empty'
        raise ValueError(msg)
    specs = [_normalize_method_spec(spec) for spec in methods]

    jl = julia()
    julia_data = to_julia_data(data)
    julia_weights = to_weights(weights)
    julia_reference = to_julia_data(reference) if reference is not None else None
    julia_reference_weights = to_weights(reference_weights)
    scoring_kernel_obj = build_kernel(jl, kernel, bandwidth)

    splitters = [_build_comparison_splitter(jl, spec, ratio, n_threads, seed) for spec in specs]
    splitters_vector = jl.Vector[jl.AbstractSplitter](splitters)

    compare_kwargs: dict[str, JuliaValue] = _estimator_kwargs(jl, estimator, seed, n_threads)
    compare_kwargs['kernel'] = scoring_kernel_obj
    compare_kwargs['exact_threshold'] = exact_threshold
    compare_kwargs.update(_weights_kwarg(julia_weights))
    compare_kwargs.update(_reference_kwargs(julia_reference, julia_reference_weights))
    compare_kwargs['standardize'] = standardize

    with _translate_error():
        comparison = jl.compare(splitters_vector, julia_data, **compare_kwargs)

    results = tuple(
        _to_split_result(jl, julia_result, spec.method, spec.kernel, ratio)
        for spec, julia_result in zip(specs, comparison.results, strict=True)
    )
    qualities = tuple(float(q) for q in comparison.qualities)
    return SplitComparison(results=results, qualities=qualities, kernel=kernel)


def _normalize_method_spec(spec: SplitMethod | Mapping[str, object]) -> _MethodSpec:
    """Normalize one entry of `compare`'s `methods` into a full `_MethodSpec`.

    Args:
        spec: A method name, or a mapping with a required ``'method'`` key
            and any of the keys in `_METHOD_OPTION_KEYS`.

    Returns:
        A `_MethodSpec` with every option set, each either the caller's
        value or its `~splitiq.split.datasplit` default.

    Raises:
        ValueError: If `spec` is neither a string nor a mapping, if a
            mapping has no ``'method'`` key, if ``spec['method']`` (or
            `spec` itself) is not a recognized method name, or if a mapping
            has a key outside ``{'method', *_METHOD_OPTION_KEYS}``.
    """
    options: Mapping[str, object]
    if isinstance(spec, str):
        method: object = spec
        options = {}
    elif isinstance(spec, Mapping):
        options = spec
        if 'method' not in options:
            msg = "each mapping in `methods` needs a 'method' key"
            raise ValueError(msg)
        method = options['method']
    else:
        msg = (
            'each entry of `methods` must be a method name or a mapping with a '
            f"'method' key, got {spec!r}"
        )
        raise ValueError(msg)
    if method not in _METHODS:
        msg = f'method must be one of {_METHODS}, got {method!r}'
        raise ValueError(msg)
    unknown = set(options) - {'method', *_METHOD_OPTION_KEYS}
    if unknown:
        msg = f'unknown option(s) {sorted(unknown)!r} for method {method!r}'
        raise ValueError(msg)
    # `options` is a caller-supplied `Mapping[str, object]`; the casts below narrow it
    # to the types `_build_splitter` (called from `_build_comparison_splitter`) expects
    # after the runtime checks above. An invalid value among them still surfaces as
    # `ValueError`, from `_build_splitter` or from Julia.
    return _MethodSpec(
        method=cast('SplitMethod', method),
        kernel=cast('SplitKernelName', options.get('kernel', 'energy')),
        bandwidth=cast('float | Literal["median"]', options.get('bandwidth', 'median')),
        kappa=cast('int | None', options.get('kappa', _DEFAULT_KAPPA)),
        max_iterations=cast('int', options.get('max_iterations', _DEFAULT_MAX_ITERATIONS)),
        tolerance=cast('float', options.get('tolerance', _DEFAULT_TOLERANCE)),
        start=cast('StartRule | None', options.get('start')),
        delta=cast('float', options.get('delta', _DEFAULT_DELTA)),
        compress=cast('CompressMode', options.get('compress', _DEFAULT_COMPRESS)),
    )


def _build_comparison_splitter(
    jl: JuliaValue,
    spec: _MethodSpec,
    ratio: float,
    n_threads: int | None,
    seed: int | None,
) -> JuliaValue:
    """Build one Julia splitter from a normalized method spec.

    Args:
        jl: The Julia ``Main`` handle from :func:`splitiq._julia.julia`.
        spec: A `_MethodSpec` from `_normalize_method_spec`.
        ratio: Fraction of rows assigned to the test set, shared by every
            splitter in the comparison.
        n_threads: Number of threads, or ``None`` to omit the keyword.
        seed: A seed to build a fresh RNG for this splitter, or ``None`` to
            omit the keyword (Julia's default RNG).

    Returns:
        A Julia ``SupportPointSplitter``, ``HerdingSplitter``,
        ``TwinningSplitter``, or ``KernelThinningSplitter`` value.

    Raises:
        ValueError: If any per-method option is invalid for its `method`
            (see `~splitiq.split._build_splitter`).
    """
    kernel_obj = build_kernel(jl, spec.kernel, spec.bandwidth)
    rng = build_rng(jl, seed)
    return _build_splitter(
        jl,
        spec.method,
        spec.kernel,
        kernel_obj,
        ratio,
        spec.kappa,
        spec.max_iterations,
        spec.tolerance,
        n_threads,
        rng,
        spec.start,
        spec.delta,
        spec.compress,
    )
