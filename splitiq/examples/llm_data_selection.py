r"""Select LLM training data from an embedding matrix with `splitiq`.

Downloads a public arXiv-abstract embedding dataset (5,000 abstracts, MiniLM
384-d; CC0), cosine-normalizes the rows, and selects `n` (default 500, ~10%)
abstracts with every splitter under three target measures (the data
itself, a quality-weighted version with abstract length as a stand-in, and
the `cs` archive as a target sub-population) against uniform random and
K-center greedy baselines. Also times Compress++ against plain kernel
thinning for a selection size much smaller than the dataset. Prints a
markdown table and writes it to `--out` when given.

Run from the `splitiq/` directory:

    uv run python examples/llm_data_selection.py

Running against a checkout (rather than the released Julia package) needs
the dev Julia project: build it once with `./scripts/setup_julia_dev.sh` (or
`make julia-dev`), then run with:

    PYTHON_JULIACALL_PROJECT=$PWD/.julia_dev PYTHON_JULIACALL_EXE=$(command -v julia) \\
        uv run python examples/llm_data_selection.py

Options: `--model minilm|arcticlarge` (default `minilm`), `--n` (default
500), `--out PATH` (write the printed table to a markdown file; nothing is
written by default), `--quick` (`n=50` over the first 1,000 rows, for a fast
smoke run).
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq

from splitiq import energydistance, select_rows

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from splitiq.split import SplitKernelName, SplitMethod

# ---------------------------------------------------------------------------
# Helpers (NumPy only, print-free).
# ---------------------------------------------------------------------------


def cosine_normalize(x: ArrayLike) -> np.ndarray:
    """L2-normalize each row of `x` to unit length.

    Args:
        x: A 2-D array-like, shape `(n, p)`.

    Returns:
        A new array of the same shape with each row divided by its L2 norm.
        A zero row (norm `0`) is returned unchanged rather than divided by
        zero.

    Raises:
        ValueError: If `x` is not 2-D.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        message = 'x must be 2-D'
        raise ValueError(message)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return x / safe_norms


def clip_quantile(values: ArrayLike, q: float) -> np.ndarray:
    """Clip `values` at their `q`-quantile, capping outliers from above.

    Used as the abstract-length quality proxy: `clip_quantile(lengths,
    0.99)` caps the longest 1% of abstracts at the 99th-percentile length.

    Args:
        values: A 1-D array-like of numeric values.
        q: Quantile in `[0, 1]` (e.g. `0.99` for the 99th percentile).

    Returns:
        `np.minimum(values, quantile(values, q))`, as a new float array.

    Raises:
        ValueError: If `q` is outside `[0, 1]`.
    """
    if not 0.0 <= q <= 1.0:
        message = 'q must be in [0, 1]'
        raise ValueError(message)
    arr = np.asarray(values, dtype=float)
    threshold = float(np.quantile(arr, q))
    return np.minimum(arr, threshold)


def has_category(categories: Sequence[str] | None, target: str) -> bool:
    """Whether a row's category list contains `target` exactly.

    Args:
        categories: A row's category list (e.g. `['cs', 'stat']`), or
            `None` for no categories.
        target: The category to look for (e.g. `'cs'`).

    Returns:
        `True` if `target` is one of `categories`, else `False`.
    """
    if categories is None:
        return False
    return target in categories


def category_mask(categories_column: Sequence[Sequence[str] | None], target: str) -> np.ndarray:
    """Boolean mask of rows whose category list contains `target`.

    Args:
        categories_column: One category list (or `None`) per row.
        target: The category to look for (e.g. `'cs'`).

    Returns:
        A boolean numpy array, one entry per row of `categories_column`.
    """
    return np.array([has_category(c, target) for c in categories_column], dtype=bool)


def kcenter_greedy(x: ArrayLike, n: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-first traversal (Sener & Savarese 2018) over rows of `x`.

    Starts from a uniformly random row, then repeatedly appends the row
    farthest (in Euclidean distance) from the closest already-selected row.

    Args:
        x: A 2-D array-like, shape `(n_rows, p)`.
        n: Number of rows to select, in `1:n_rows`.
        rng: Random number generator; only its starting draw is random, the
            rest of the traversal is deterministic given the start.

    Returns:
        A 0-based integer array of `n` distinct row indices, in selection
        order (the random start first).

    Raises:
        ValueError: If `n` is not in `1:n_rows`.
    """
    x = np.asarray(x, dtype=float)
    n_rows = x.shape[0]
    if not 1 <= n <= n_rows:
        message = f'n must be in 1:{n_rows}, got {n}'
        raise ValueError(message)
    selected = [int(rng.integers(n_rows))]
    min_dist = np.full(n_rows, np.inf)
    for _ in range(1, n):
        last = x[selected[-1]]
        dist_to_last = np.linalg.norm(x - last, axis=1)
        min_dist = np.minimum(min_dist, dist_to_last)
        selected.append(int(np.argmax(min_dist)))
    return np.array(selected, dtype=np.int64)


def build_embedding_matrix(embedding_rows: Sequence[Sequence[float] | None], p: int) -> np.ndarray:
    """Stack per-row embedding lists into a dense matrix.

    A missing row, or a missing component within a row, is filled with
    `0.0` (matching the Julia example's `coalesce(e, 0.0)`).

    Args:
        embedding_rows: One embedding (or `None`) per row, each of length
            `p` when present.
        p: Embedding dimensionality.

    Returns:
        A `(len(embedding_rows), p)` float64 array.
    """
    matrix = np.zeros((len(embedding_rows), p), dtype=float)
    for i, row in enumerate(embedding_rows):
        if row is None:
            continue
        for j, value in enumerate(row):
            matrix[i, j] = 0.0 if value is None else float(value)
    return matrix


def content_lengths(content_column: Sequence[str]) -> np.ndarray:
    """Character length of each abstract in `content_column`.

    Args:
        content_column: One abstract string per row.

    Returns:
        A float array of `len(text)` for each entry.
    """
    return np.array([len(text) for text in content_column], dtype=float)


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

_DATASET_URL_TEMPLATE = (
    'https://huggingface.co/datasets/sondalex/arxiv-abstracts-2021-embeddings-10000/'
    'resolve/main/data/arxiv-abstract-{model}.parquet'
)
_QUICK_ROWS = 1_000


def _download_dataset(model: str, dest: Path) -> None:
    """Download the arXiv-abstract embedding parquet for `model`, unless cached.

    Args:
        model: `'minilm'` or `'arcticlarge'`.
        dest: Local file path to save to. The download is skipped when this
            file already exists.
    """
    if dest.exists():
        return
    url = _DATASET_URL_TEMPLATE.format(model=model)
    if not url.startswith('https://huggingface.co/'):  # pragma: no cover - defensive
        message = f'unexpected dataset URL: {url}'
        raise ValueError(message)
    print(f'downloading {url}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 -- fixed https://huggingface.co host


def _load_data(model: str, *, quick: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download (if needed), read, and preprocess the embedding dataset.

    Args:
        model: `'minilm'` or `'arcticlarge'`.
        quick: If `True`, use only the first `_QUICK_ROWS` rows.

    Returns:
        `(e, w, is_cs)`: `e` is the cosine-normalized embedding matrix, `w`
        is the length-based quality proxy (clipped at the 99th percentile),
        and `is_cs` is a boolean mask of rows in the `cs` archive.
    """
    data_dir = Path(__file__).resolve().parent / 'data'
    dest = data_dir / f'arxiv-abstract-{model}.parquet'
    _download_dataset(model, dest)

    table = pq.read_table(dest, columns=['categories', 'content', 'embedding'])
    if quick:
        table = table.slice(0, min(_QUICK_ROWS, table.num_rows))
    categories_column = table.column('categories').to_pylist()
    content_column = table.column('content').to_pylist()
    embedding_column = table.column('embedding').to_pylist()

    p = len(next(row for row in embedding_column if row is not None))
    e = cosine_normalize(build_embedding_matrix(embedding_column, p))
    w = clip_quantile(content_lengths(content_column), 0.99)
    is_cs = category_mask(categories_column, 'cs')
    return e, w, is_cs


# ---------------------------------------------------------------------------
# Scoring and timed selection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreRow:
    """One row of the results table."""

    setting: str
    method: str
    optimized: float
    plain: float
    seconds: float | None


def _score_plain(e: np.ndarray, sel: np.ndarray) -> float:
    return float(energydistance(e[sel], e))


def _scorer_for_setting(
    setting: str, e: np.ndarray, w: np.ndarray, r: np.ndarray
) -> Callable[[np.ndarray], float]:
    """The energy-distance scorer for `setting`'s optimized measure.

    Args:
        setting: `'plain'`, `'weights = length'`, or `'reference = cs'`.
        e: The full (cosine-normalized) embedding matrix.
        w: The length-based quality weights, one per row of `e`.
        r: The `cs`-archive reference rows.

    Returns:
        A callable mapping a selection's row indices to its energy distance
        to `setting`'s target measure.

    Raises:
        ValueError: If `setting` is not recognized.
    """
    if setting == 'plain':
        return lambda sel: float(energydistance(e[sel], e))
    if setting == 'weights = length':
        return lambda sel: float(energydistance(e[sel], e, weights_y=w))
    if setting == 'reference = cs':
        return lambda sel: float(energydistance(e[sel], r))
    message = f'unknown setting: {setting}'
    raise ValueError(message)


def _random_selection(n_rows: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(n_rows, size=n, replace=False)


def _timed_select(
    e: np.ndarray,
    n: int,
    *,
    method: SplitMethod,
    kernel: SplitKernelName | None,
    warmup_seed: int | None,
    timed_seed: int | None,
    extra_kwargs: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Time one `select_rows` call after a warm-up on a 200-row slice.

    The warm-up uses a distinct seed and a 200-row prefix of `e`, so the
    timed call's own random stream is untouched. A `weights` entry of
    `extra_kwargs` is sliced to match the warm-up subset; `reference` is
    passed through unchanged (it does not depend on the number of candidate
    rows).

    Args:
        e: Candidate rows, shape `(n_rows, p)`.
        n: Number of rows to select for the timed call.
        method: `select_rows`'s `method` keyword.
        kernel: `select_rows`'s `kernel` keyword, or `None` to omit it
            (for `method='twinning'`, which has a fixed kernel).
        warmup_seed: Seed for the warm-up call.
        timed_seed: Seed for the timed call.
        extra_kwargs: Extra `select_rows` keyword arguments (e.g.
            `weights`, `reference`, `kappa`, `max_iterations`).

    Returns:
        `(selection, seconds)`: the timed call's 0-based row indices and
        its wall-clock duration.
    """
    warmup_rows = min(200, e.shape[0])
    warmup_kwargs = dict(extra_kwargs)
    if 'weights' in warmup_kwargs:
        warmup_kwargs['weights'] = warmup_kwargs['weights'][:warmup_rows]
    common: dict[str, Any] = {'standardize': False}
    if kernel is not None:
        common['kernel'] = kernel

    select_rows(
        e[:warmup_rows],
        min(20, warmup_rows),
        method=method,
        seed=warmup_seed,
        **common,
        **warmup_kwargs,
    )
    start = time.perf_counter()
    selection = select_rows(e, n, method=method, seed=timed_seed, **common, **extra_kwargs)
    seconds = time.perf_counter() - start
    return selection, seconds


# ---------------------------------------------------------------------------
# Settings, splitters, and the main comparison.
# ---------------------------------------------------------------------------

# (label, method, kernel or None, extra select_rows kwargs)
_SPLITTERS: tuple[tuple[str, SplitMethod, SplitKernelName | None, dict[str, Any]], ...] = (
    ('herding · energy', 'herding', 'energy', {}),
    ('twinning', 'twinning', None, {}),
    ('kernel thinning · energy', 'kernel_thinning', 'energy', {}),
    (
        'support points · energy',
        'support_points',
        'energy',
        {'kappa': 1_000, 'max_iterations': 100},
    ),
)


def _settings(
    w: np.ndarray, r: np.ndarray
) -> tuple[tuple[str, dict[str, Any], tuple[str, ...]], ...]:
    """The three settings (target measures) the example compares, in order.

    Args:
        w: The length-based quality weights.
        r: The `cs`-archive reference rows.

    Returns:
        `(label, extra select_rows kwargs, methods to skip)` triples.
        Twinning is skipped under `weights`/`reference` because it rejects
        both.
    """
    return (
        ('plain', {}, ()),
        ('weights = length', {'weights': w}, ('twinning',)),
        ('reference = cs', {'reference': r}, ('twinning',)),
    )


def _run_setting(
    e: np.ndarray,
    n: int,
    setting: str,
    setting_kwargs: dict[str, Any],
    skip: tuple[str, ...],
    scorer: Callable[[np.ndarray], float],
) -> list[ScoreRow]:
    """Run the random and k-center baselines plus every non-skipped splitter for one setting."""
    rows: list[ScoreRow] = []

    randoms = [_random_selection(e.shape[0], n, seed) for seed in range(101, 106)]
    rows.append(
        ScoreRow(
            setting,
            'random',
            float(np.mean([scorer(sel) for sel in randoms])),
            float(np.mean([_score_plain(e, sel) for sel in randoms])),
            None,
        )
    )

    warmup_rows = min(200, e.shape[0])
    kcenter_greedy(e[:warmup_rows], min(20, warmup_rows), np.random.default_rng(0))
    start = time.perf_counter()
    kcenter_sel = kcenter_greedy(e, n, np.random.default_rng(7))
    kcenter_seconds = time.perf_counter() - start
    rows.append(
        ScoreRow(
            setting,
            'k-center greedy',
            scorer(kcenter_sel),
            _score_plain(e, kcenter_sel),
            kcenter_seconds,
        )
    )

    for label, method, kernel, extra in _SPLITTERS:
        if label in skip:
            continue
        sel, seconds = _timed_select(
            e,
            n,
            method=method,
            kernel=kernel,
            warmup_seed=0,
            timed_seed=1,
            extra_kwargs={**extra, **setting_kwargs},
        )
        rows.append(ScoreRow(setting, label, scorer(sel), _score_plain(e, sel), seconds))

    return rows


def _run_compress_section(e: np.ndarray, n: int) -> list[ScoreRow]:
    """Compare Compress++ (`compress='always'`) against plain kernel thinning at `n` << N."""
    setting = f'plain, n = {n}'
    rows: list[ScoreRow] = []
    for label, mode in (
        ("kernel thinning · compress='never'", 'never'),
        ("kernel thinning · compress='always'", 'always'),
    ):
        warmup_rows = min(400, e.shape[0])
        select_rows(
            e[:warmup_rows],
            min(20, warmup_rows),
            method='kernel_thinning',
            kernel='energy',
            compress=mode,
            seed=0,
            standardize=False,
        )
        start = time.perf_counter()
        sel = select_rows(
            e,
            n,
            method='kernel_thinning',
            kernel='energy',
            compress=mode,
            seed=3,
            standardize=False,
        )
        seconds = time.perf_counter() - start
        plain = _score_plain(e, sel)
        rows.append(ScoreRow(setting, label, plain, plain, seconds))

    randoms = [_random_selection(e.shape[0], n, seed) for seed in range(201, 206)]
    plain_mean = float(np.mean([_score_plain(e, sel) for sel in randoms]))
    rows.append(ScoreRow(setting, 'random', plain_mean, plain_mean, None))
    return rows


def _rows_to_markdown(rows: list[ScoreRow]) -> str:
    header = (
        '| setting | method | energy distance to the optimized measure '
        '| energy distance to the data | seconds |'
    )
    sep = '|---|---|---:|---:|---:|'
    lines = [header, sep]
    for row in rows:
        seconds = '-' if row.seconds is None else f'{row.seconds:.2g}'
        lines.append(
            f'| {row.setting} | {row.method} | {row.optimized:.3g} | {row.plain:.3g} | {seconds} |'
        )
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

_QUICK_N = 50
_COMPRESS_N = 250
_QUICK_COMPRESS_N = 50


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model', choices=['minilm', 'arcticlarge'], default='minilm', help='Embedding model.'
    )
    parser.add_argument('--n', type=int, default=500, help='Number of rows to select.')
    parser.add_argument(
        '--out', type=Path, default=None, help='Write the printed table to this markdown file.'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use a smaller run (n=50, first 1,000 rows) for a fast smoke run.',
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the LLM data-selection comparison and print (optionally write) the results table."""
    args = _parse_args()
    e, w, is_cs = _load_data(args.model, quick=args.quick)
    r = e[is_cs]
    n_rows, p = e.shape
    n_select = _QUICK_N if args.quick else args.n
    print(f'N = {n_rows}, p = {p}, target rows (cs) = {r.shape[0]}, n = {n_select}')

    rows: list[ScoreRow] = []
    for setting, setting_kwargs, skip in _settings(w, r):
        scorer = _scorer_for_setting(setting, e, w, r)
        rows.extend(_run_setting(e, n_select, setting, setting_kwargs, skip, scorer))

    compress_n = _QUICK_COMPRESS_N if args.quick else _COMPRESS_N
    rows.extend(_run_compress_section(e, compress_n))

    table = _rows_to_markdown(rows)
    print()
    print(table)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
