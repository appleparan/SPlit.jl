r"""Flatten time-series windows into rows and select a distribution-preserving subset.

This example turns a multivariate time series into fixed-length, non-overlapping
windows (segments), flattens each window into one row in variable-major order,
and runs `splitiq` selectors (twinning, herding, kernel thinning, support points)
against uniform random baselines on a synthetic two-regime series. It shows:

- the window-flattening rule and how to recover the original rows of a selected
  window from its 0-based start index;
- that windows sharing the same point-level mean/variance can still be told
  apart at the window level, because their *within-window* structure differs;
- that a random split of a time series interleaves train and test in time, so
  `datasplit` on windows measures interpolation, not forecasting;
- contrast 1: what happens when the window length is shorter than the series'
  dependence length (the selector's advantage over random should vanish),
  averaged over several independently generated datasets.

Run from the `splitiq/` directory:

    uv run python examples/time_series_windows.py

Running against a checkout (rather than the released Julia package) needs the
dev Julia project: build it once with `./scripts/setup_julia_dev.sh` (or
`make julia-dev`), then run with:

    PYTHON_JULIACALL_PROJECT=$PWD/.julia_dev PYTHON_JULIACALL_EXE=$(command -v julia) \\
        uv run python examples/time_series_windows.py

Options: `--out PATH` writes the printed tables as one markdown file (nothing is
written by default); `--quick` shrinks the synthetic demo (`m=200`, `n=20`
instead of `m=1000`, `n=100`) for a fast smoke run.

Source on the repository main branch:
https://github.com/appleparan/SPlit.jl/blob/main/splitiq/examples/time_series_windows.py
Julia counterpart: https://github.com/appleparan/SPlit.jl/blob/main/examples/time_series_windows.jl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from splitiq import datasplit, energydistance, select_rows

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Helpers (NumPy only, print-free).
# ---------------------------------------------------------------------------


def window_count(n_rows: int, length: int, stride: int) -> int:
    """Number of non-overlapping-by-`stride` windows a series of `n_rows` yields.

    Args:
        n_rows: Number of rows (time steps) in the series.
        length: Window length, in rows. Must be positive.
        stride: Step between consecutive window starts, in rows. Must be
            positive.

    Returns:
        `0` if `n_rows < length` (only a trailing partial window would fit,
        and it is dropped); otherwise `(n_rows - length) // stride + 1`.

    Raises:
        ValueError: If `length` or `stride` is not positive.
    """
    if length <= 0:
        message = 'length must be positive'
        raise ValueError(message)
    if stride <= 0:
        message = 'stride must be positive'
        raise ValueError(message)
    if n_rows < length:
        return 0
    return (n_rows - length) // stride + 1


def windows(x: ArrayLike, length: int, stride: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Flatten fixed-length windows of `x` into rows, variable-major.

    Window `i` covers rows `starts[i]:starts[i] + length` of `x` and is
    flattened as `x[s:s + length].reshape(-1, order='F')`: all `length`
    values of variable 0, then all `length` values of variable 1, and so on.
    Variable `v` at offset `t` (both 0-based) lands in column `v * length + t`
    of the result.

    Args:
        x: A 2-D array-like, shape `(n_rows, p)` (rows are time steps,
            columns are variables). A 1-D array is treated as one variable.
        length: Window length, in rows. Must be positive.
        stride: Step between consecutive window starts, in rows. Defaults to
            `length` (non-overlapping windows). Must be positive.

    Returns:
        `(z, starts)`: `z` has shape `(m, length * p)` where `m =
        window_count(n_rows, length, stride)`; `starts` is a 0-based integer
        array of the `m` window start rows. Both are empty (`z` has 0 rows,
        `starts` has 0 elements) when `n_rows < length`.

    Raises:
        ValueError: If `length` or `stride` is not positive.
    """
    if stride is None:
        stride = length
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n_rows, p = x.shape
    m = window_count(n_rows, length, stride)
    starts = (np.arange(m, dtype=np.int64) * stride).astype(np.int64)
    z = np.empty((m, length * p), dtype=float)
    for i, start in enumerate(starts):
        z[i, :] = x[start : start + length].reshape(-1, order='F')
    return z, starts


def recover_window(x: np.ndarray, start: int, length: int) -> np.ndarray:
    """The original `(length, p)` slice of `x` a flattened window came from.

    Args:
        x: The 2-D array `windows` was built from.
        start: 0-based start row of the window (an entry of `windows`'
            `starts`).
        length: Window length, in rows.

    Returns:
        `x[start:start + length]`.
    """
    x = np.asarray(x)
    return x[start : start + length]


def standardize_by_variable(
    z: np.ndarray,
    length: int,
    p: int,
    fit: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Standardize flattened windows per variable, sharing scale across offsets.

    For each variable `v`, all `length` offsets of that variable (columns
    `v * length : (v + 1) * length`) are centered and scaled by one mean and
    one standard deviation, computed over every window and every offset of
    that variable combined.

    Args:
        z: Flattened windows, shape `(m, length * p)`, variable-major (as
            returned by `windows`).
        length: Window length used to build `z`.
        p: Number of variables.
        fit: `(means, stds)` from a previous call (each of length `p`), to
            apply the same transform to a different batch of windows. `None`
            fits the mean/std on `z` itself.

    Returns:
        `(zs, fit)`: the standardized windows and the `(means, stds)` used,
        so the same transform can be re-applied to other windows.
    """
    z = np.asarray(z, dtype=float)
    if fit is None:
        means = np.empty(p, dtype=float)
        stds = np.empty(p, dtype=float)
        for v in range(p):
            block = z[:, v * length : (v + 1) * length]
            means[v] = block.mean()
            std = block.std(ddof=1)
            stds[v] = std if std > 0.0 else 1.0
        fit = (means, stds)
    means, stds = fit
    zs = np.empty_like(z)
    for v in range(p):
        sl = slice(v * length, (v + 1) * length)
        zs[:, sl] = (z[:, sl] - means[v]) / stds[v]
    return zs, fit


def lag1_autocorrelation(z_row: np.ndarray, length: int, p: int) -> float:
    """Mean lag-1 sample autocorrelation of one flattened window, over variables.

    Args:
        z_row: One flattened window, shape `(length * p,)`, variable-major.
        length: Window length.
        p: Number of variables.

    Returns:
        The average, over variables, of the lag-1 sample autocorrelation of
        that variable's `length`-long series within the window. A constant
        variable (zero variance) contributes `0.0`.
    """
    z_row = np.asarray(z_row, dtype=float)
    autocorrelations = np.empty(p, dtype=float)
    for v in range(p):
        y = z_row[v * length : (v + 1) * length]
        deviation = y - y.mean()
        denominator = float(np.sum(deviation**2))
        if denominator == 0.0:
            autocorrelations[v] = 0.0
            continue
        numerator = float(np.sum(deviation[:-1] * deviation[1:]))
        autocorrelations[v] = numerator / denominator
    return float(autocorrelations.mean())


def two_regime_series(
    rng: np.random.Generator,
    m: int,
    length: int,
    p: int = 3,
    share_a: float = 0.7,
    stay_a: float = 0.94,
    stay_b: float = 0.10,
    mu: tuple[float, ...] = (1.0, 0.7, 1.3),
    sigma: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate `m` windows of length `length` from two regimes with equal point marginals.

    Each window is one draw of a two-state Markov chain `s_t in {-1, +1}`
    (starting at +-1 with equal probability), shared across variables within
    that window. Variable `v` at offset `t` is `s_t * a * mu[v] + sigma *
    eps_t`, with `eps_t` standard normal noise and `a` a per-window amplitude
    factor drawn from `Uniform(0.8, 1.2)`. Regime A ("persistent") uses stay
    probability `stay_a`; regime B ("alternating") uses `stay_b`. Because `s`
    is symmetric around zero regardless of regime, the point-level mean and
    variance of every variable coincide across regimes (`mu[v]**2 + sigma**2`);
    only the within-window (temporal) structure differs.

    Args:
        rng: Random number generator; all randomness is drawn from it.
        m: Number of windows.
        length: Window length, in rows.
        p: Number of variables.
        share_a: Fraction of windows assigned to regime A.
        stay_a: Regime A's Markov-chain stay probability.
        stay_b: Regime B's Markov-chain stay probability.
        mu: Per-variable base amplitude, one entry per variable.
        sigma: Noise standard deviation.

    Returns:
        `(x, labels)`: `x` has shape `(m * length, p)`; `labels` is a length-`m`
        array of `'A'`/`'B'` regime labels, one per window.

    Raises:
        ValueError: If `m`, `length`, or `p` is not positive, or if `mu` has
            fewer than `p` entries.
    """
    if m <= 0:
        message = 'm must be positive'
        raise ValueError(message)
    if length <= 0:
        message = 'length must be positive'
        raise ValueError(message)
    if p <= 0:
        message = 'p must be positive'
        raise ValueError(message)
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.size < p:
        message = 'mu must have at least p entries'
        raise ValueError(message)
    is_a = rng.random(m) < share_a
    x = np.empty((m * length, p), dtype=float)
    labels = np.empty(m, dtype='<U1')
    for i in range(m):
        labels[i] = 'A' if is_a[i] else 'B'
        stay_prob = stay_a if is_a[i] else stay_b
        s = np.empty(length, dtype=float)
        s[0] = 1.0 if rng.random() < 0.5 else -1.0
        if length > 1:
            flips = rng.random(length - 1) >= stay_prob
            for t in range(1, length):
                s[t] = -s[t - 1] if flips[t - 1] else s[t - 1]
        amplitude = rng.uniform(0.8, 1.2)
        eps = rng.standard_normal((length, p))
        window = np.empty((length, p), dtype=float)
        for v in range(p):
            mu_v = mu_arr[v]
            window[:, v] = s * amplitude * mu_v + sigma * eps[:, v]
        x[i * length : (i + 1) * length, :] = window
    return x, labels


# ---------------------------------------------------------------------------
# Reporting helpers (not covered by tests; formatting only).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionMetrics:
    """Quality metrics for one selection of windows."""

    energy_distance: float
    regime_error: float
    lag1_mean: float


def _evaluate_selection(
    zs: np.ndarray,
    labels: np.ndarray,
    sel: np.ndarray,
    length: int,
    p: int,
    share_a: float,
) -> SelectionMetrics:
    energy = float(energydistance(zs[sel], zs))
    regime_error = float(abs(np.mean(labels[sel] == 'A') - share_a))
    lag1_mean = float(np.mean([lag1_autocorrelation(zs[i], length, p) for i in sel]))
    return SelectionMetrics(energy, regime_error, lag1_mean)


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, sd


def _fmt(mean: float, sd: float) -> str:
    return f'{mean:.4f} +/- {sd:.4f}' if sd > 0.0 else f'{mean:.4f}'


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = '| ' + ' | '.join(headers) + ' |'
    sep_line = '| ' + ' | '.join('---' for _ in headers) + ' |'
    row_lines = ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join([header_line, sep_line, *row_lines])


def _print_and_collect(title: str, table: str, sections: list[str]) -> None:
    print(f'\n## {title}\n')
    print(table)
    sections.append(f'## {title}\n\n{table}\n')


# ---------------------------------------------------------------------------
# Demo sections.
# ---------------------------------------------------------------------------


def _demo_fixture(sections: list[str]) -> None:
    x = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]
    length, stride = 2, 2
    z, starts = windows(x, length, stride)
    print('x (N=5 rows, p=2 variables):')
    print(np.array(x))
    print(f'\nlength={length}, stride={stride} -> window_count = {window_count(5, length, stride)}')
    print(
        f'starts (0-based) = {starts.tolist()}  (row index 4 is a trailing partial window, dropped)'
    )
    print('\nz (variable-major: variable v, offset t -> column v*length + t, both 0-based):')
    print(z)
    table = _markdown_table(
        ['row', 'z'],
        [[str(i), str(row.tolist())] for i, row in enumerate(z)],
    )
    sections.append(f'## Fixture\n\nstarts = {starts.tolist()}\n\n{table}\n')


def _demo_point_vs_window(
    x: np.ndarray,
    labels: np.ndarray,
    zs: np.ndarray,
    length: int,
    p: int,
    sections: list[str],
) -> None:
    point_labels = np.repeat(labels, length)
    mask_a, mask_b = point_labels == 'A', point_labels == 'B'
    win_a, win_b = labels == 'A', labels == 'B'

    rows = []
    for v in range(p):
        mean_a, var_a = x[mask_a, v].mean(), x[mask_a, v].var()
        mean_b, var_b = x[mask_b, v].mean(), x[mask_b, v].var()
        rows.append([str(v), f'{mean_a:.4f}', f'{var_a:.4f}', f'{mean_b:.4f}', f'{var_b:.4f}'])
    table = _markdown_table(['variable', 'mean A', 'var A', 'mean B', 'var B'], rows)
    _print_and_collect('Point-level moments, regime A vs B', table, sections)

    ed_points = energydistance(x[mask_a], x[mask_b])
    ed_windows = energydistance(zs[win_a], zs[win_b])
    table = _markdown_table(
        ['level', 'energy distance A vs B'],
        [['point', f'{ed_points:.4f}'], ['window', f'{ed_windows:.4f}']],
    )
    _print_and_collect('Energy distance, A vs B: point level vs window level', table, sections)


def _select_twinning(zs: np.ndarray, n: int, _seed: int | None) -> np.ndarray:
    return select_rows(zs, n, method='twinning', standardize=False)


def _select_herding(zs: np.ndarray, n: int, _seed: int | None) -> np.ndarray:
    return select_rows(zs, n, method='herding', kernel='energy', standardize=False)


def _select_kernel_thinning(zs: np.ndarray, n: int, seed: int | None) -> np.ndarray:
    return select_rows(zs, n, method='kernel_thinning', standardize=False, seed=seed)


def _select_support_points(zs: np.ndarray, n: int, seed: int | None) -> np.ndarray:
    return select_rows(
        zs, n, method='support_points', kappa=300, max_iterations=100, standardize=False, seed=seed
    )


def _selector_specs() -> list[
    tuple[str, Callable[[np.ndarray, int, int | None], np.ndarray], list[int | None]]
]:
    return [
        ('twinning', _select_twinning, [None]),
        ('herding', _select_herding, [None]),
        ('kernel_thinning', _select_kernel_thinning, [1, 2, 3]),
        ('support_points', _select_support_points, [1, 2, 3]),
    ]


def _random_baseline(
    zs: np.ndarray, labels: np.ndarray, length: int, p: int, n: int, share_a: float, seeds: range
) -> list[SelectionMetrics]:
    m = zs.shape[0]
    metrics = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sel = rng.choice(m, size=n, replace=False)
        metrics.append(_evaluate_selection(zs, labels, sel, length, p, share_a))
    return metrics


def _row_from_metrics(name: str, metrics: list[SelectionMetrics]) -> list[str]:
    energy_mean, energy_sd = _mean_sd([mm.energy_distance for mm in metrics])
    regime_mean, regime_sd = _mean_sd([mm.regime_error for mm in metrics])
    lag1_mean, lag1_sd = _mean_sd([mm.lag1_mean for mm in metrics])
    return [
        name,
        _fmt(energy_mean, energy_sd),
        _fmt(regime_mean, regime_sd),
        _fmt(lag1_mean, lag1_sd),
    ]


@dataclass(frozen=True)
class ComparisonResult:
    """Output of `_demo_selector_comparison`, for reuse in later sections."""

    twinning_selection: np.ndarray


def _demo_selector_comparison(
    zs: np.ndarray,
    labels: np.ndarray,
    length: int,
    p: int,
    n: int,
    share_a: float,
    sections: list[str],
) -> ComparisonResult:
    m = zs.shape[0]
    rows: list[list[str]] = []
    twinning_selection: np.ndarray | None = None

    for name, selector_fn, seeds in _selector_specs():
        metrics_by_seed = []
        for seed in seeds:
            sel = selector_fn(zs, n, seed)
            if name == 'twinning':
                twinning_selection = sel
            metrics_by_seed.append(_evaluate_selection(zs, labels, sel, length, p, share_a))
        rows.append(_row_from_metrics(name, metrics_by_seed))

    random_metrics = _random_baseline(zs, labels, length, p, n, share_a, range(1, 21))
    rows.append(_row_from_metrics('random (20 draws)', random_metrics))

    full_lag1 = float(np.mean([lag1_autocorrelation(zs[i], length, p) for i in range(m)]))
    rows.append(['full set', '0.0000', '0.0000', f'{full_lag1:.4f}'])

    table = _markdown_table(
        [
            'selector',
            'energy distance to full set',
            'regime-proportion error',
            'mean lag-1 autocorr',
        ],
        rows,
    )
    _print_and_collect(f'Selector comparison (n={n} of {m} windows)', table, sections)

    if twinning_selection is None:  # pragma: no cover - defensive
        message = 'twinning selection was not computed'
        raise RuntimeError(message)
    return ComparisonResult(twinning_selection)


def _demo_datasplit_and_recovery(
    zs: np.ndarray, x: np.ndarray, starts: np.ndarray, length: int, twinning_selection: np.ndarray
) -> None:
    # This split is not chronological: twinning picks representative windows
    # throughout the series, so train and test interleave in time. It measures
    # how well the test windows are interpolated from the rest of the series,
    # not whether a model can forecast ahead of them.
    result = datasplit(zs, ratio=0.2, method='twinning', standardize=False)
    print(
        f'\ndatasplit: {len(result.train_indices)} train / {len(result.test_indices)} test windows'
    )

    print('\nRecovered windows for 3 twinning selections (kept separate, never concatenated):')
    for idx in twinning_selection[:3]:
        recovered = recover_window(x, int(starts[idx]), length)
        print(f'  window {int(idx)} (start={int(starts[idx])}): shape={recovered.shape}')


def _demo_contrast_1(
    m: int,
    length: int,
    p: int,
    n: int,
    share_a: float,
    data_seeds: int,
    sections: list[str],
) -> None:
    # Twinning is deterministic, so run-to-run noise in the ratio to random
    # comes from the dataset, not the selector: average over `data_seeds`
    # independently generated datasets rather than judging L_short on a
    # single series (the main section's `zs` is left untouched).
    length_shorts = [
        length_short for length_short in (1, 2, 4, 8, 16, 32) if length_short <= length
    ]
    ratios: list[list[float]] = [[] for _ in length_shorts]
    regime_errors: list[list[float]] = [[] for _ in length_shorts]

    for d in range(1, data_seeds + 1):
        rng_d = np.random.default_rng(500 + d)
        x_d, labels_d = two_regime_series(rng_d, m, length, p=p, share_a=share_a)
        z_d, _starts_d = windows(x_d, length, length)
        zs_d, _fit_d = standardize_by_variable(z_d, length, p)

        random_energies = []
        for k in range(1, 21):
            rng_k = np.random.default_rng(1000 + 100 * d + k)
            sel_k = rng_k.choice(m, size=n, replace=False)
            random_energies.append(float(energydistance(zs_d[sel_k], zs_d)))
        random_energy_mean_d = float(np.mean(random_energies))

        for i, length_short in enumerate(length_shorts):
            # Same segment starts as the full-length windowing (stride=length).
            z_short, _starts_short = windows(x_d, length_short, length)
            zs_short, _fit = standardize_by_variable(z_short, length_short, p)
            sel = select_rows(zs_short, n, method='twinning', standardize=False)
            # Evaluate in this dataset's full-length standardized space.
            energy = float(energydistance(zs_d[sel], zs_d))
            ratio = energy / random_energy_mean_d if random_energy_mean_d > 0.0 else float('nan')
            regime_error = float(abs(np.mean(labels_d[sel] == 'A') - share_a))
            ratios[i].append(ratio)
            regime_errors[i].append(regime_error)

    rows = []
    for length_short, ratio_values, regime_values in zip(
        length_shorts, ratios, regime_errors, strict=True
    ):
        ratio_mean, ratio_sd = _mean_sd(ratio_values)
        regime_mean, regime_sd = _mean_sd(regime_values)
        rows.append([str(length_short), _fmt(ratio_mean, ratio_sd), _fmt(regime_mean, regime_sd)])

    table = _markdown_table(['L_short', 'ratio to random mean', 'regime-proportion error'], rows)
    _print_and_collect(
        'Contrast 1: representation length below the dependence length '
        f'(mean +/- sd over {data_seeds} data seeds)',
        table,
        sections,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out', type=Path, default=None, help='Write the printed tables to this markdown file.'
    )
    parser.add_argument(
        '--quick', action='store_true', help='Use a smaller demo (m=200, n=20) for a fast run.'
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the fixture printout, synthetic demo, selector comparison, and contrast 1."""
    args = _parse_args()
    sections: list[str] = []

    print('=' * 78)
    print('Fixture')
    print('=' * 78)
    _demo_fixture(sections)

    length, p = 32, 3
    m, n = (200, 20) if args.quick else (1000, 100)
    share_a = 0.7

    print('\n' + '=' * 78)
    print(f'Synthetic two-regime series (m={m}, length={length}, p={p}, n={n})')
    print('=' * 78)
    rng = np.random.default_rng(0)
    x, labels = two_regime_series(rng, m, length, p=p, share_a=share_a)
    z, starts = windows(x, length, length)
    zs, _fit = standardize_by_variable(z, length, p)

    _demo_point_vs_window(x, labels, zs, length, p, sections)
    comparison = _demo_selector_comparison(zs, labels, length, p, n, share_a, sections)
    _demo_datasplit_and_recovery(zs, x, starts, length, comparison.twinning_selection)

    print('\n' + '=' * 78)
    print('Contrast 1')
    print('=' * 78)
    data_seeds = 2 if args.quick else 5
    _demo_contrast_1(m, length, p, n, share_a, data_seeds, sections)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text('\n'.join(sections) + '\n')
        print(f'\nWrote tables to {args.out}')


if __name__ == '__main__':
    main()
