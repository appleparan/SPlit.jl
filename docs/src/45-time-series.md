# [Time-series windows](@id time-series)

The splitters in this package select rows. A time series is not a table of
rows: turning it into one means cutting it into fixed-length windows and
flattening each window into one row. That preserves the distribution of
windows and nothing beyond the window length. Past a few hundred columns
the selectors' advantage over random shrinks and compilation time grows, so
long windows need a different representation. The numbers on this page
come from `examples/time_series_windows.jl` and its Python counterpart,
`splitiq/examples/time_series_windows.py`.

## Point level and window level

Matching a selection's per-point mean and variance to the full series does
not mean the selection captures the series' temporal structure. Two regimes
can share identical point-level moments while differing entirely in how
consecutive points depend on each other — a persistent regime and an
alternating one, say, built so every per-variable mean and variance line up.

On the example's two-regime series (`M = 1000` windows, `L = 32`, `p = 3`),
the energy distance between regime A and regime B is 0.00175 at the point
level and 0.529 once each window is flattened and standardized. The
difference the regimes carry is invisible until a whole window becomes the
unit of comparison.

## From series to rows

The shape chain is ``X`` (``N \times p``, one row per time step) to
``W_i`` (``L \times p``, one window) to ``Z`` (``M \times Lp``, one row per
window). Windows start at rows ``1, 1+\text{stride}, 1+2\,\text{stride},
\dots``; ``M = \lfloor (N-L)/\text{stride} \rfloor + 1`` when ``N \ge L``,
zero windows when ``N < L`` (not an error), and any trailing partial window
is dropped rather than padded.

Flattening is variable-major: row ``i`` of ``Z`` holds all ``L`` offsets of
variable 1, then all ``L`` offsets of variable 2, and so on. Variable ``v``
at offset ``t`` sits in column ``v \cdot L + t`` (0-based) or
``(v-1) \cdot L + t`` (1-based).

The shared fixture, ``X`` a 5-row, 2-variable series with `L = 2`,
`stride = 2`:

```text
X = [1 10; 2 20; 3 30; 4 40; 5 50]
Z = [1 2 10 20;
     3 4 30 40]
```

Row 5 does not fill a full window and is dropped. Julia's `starts` are
`[1, 3]` (1-based); Python's are `[0, 2]` (0-based). Recovery undoes the
flattening exactly:

```julia
reshape(Z[i, :], L, p) == X[starts[i]:starts[i]+L-1, :]
```

```python
z[i].reshape(length, p, order="F") == x[starts[i]:starts[i] + length]
```

These helpers (`windows`, `recover_window`, `standardize_by_variable`,
`two_regime_series`, `lag1_autocorrelation`) live in
`examples/time_series_windows_helpers.jl` and
`splitiq/examples/time_series_windows.py`. They are example-local, not part
of the package.

## Workflow

1. Build non-overlapping windows with `stride = L`. Overlapping windows
   ("Boundaries" below) are not independent samples.
2. Standardize per variable, not per column: all `L` offsets of one
   variable are the same physical quantity at different lags and should
   share one mean and scale. Fit on the training block when there is a
   chronological holdout; never fit per window.
3. Select with `standardize = false` — the flattened windows are already on
   the scale you want, and passing a raw numeric matrix through unchanged is
   what that keyword is for (see [Methods](@ref methods)).
4. Recover the original `L x p` slice for each selected window and keep it
   as its own sample; selected windows are never concatenated.

```julia
using SPlit, Random
include("time_series_windows_helpers.jl")

Z, starts = windows(X, L; stride = L)
Zs = standardize_by_variable(Z, L, p)
idx = selectrows(TwinningSplitter(), Zs, n; standardize = false)
recovered = [recover_window(X, starts[i], L) for i in idx]
```

```python
from time_series_windows import windows, standardize_by_variable, recover_window
from splitiq import select_rows

z, starts = windows(x, length, stride=length)
zs, _fit = standardize_by_variable(z, length, p)
idx = select_rows(zs, n, method="twinning", standardize=False)
recovered = [recover_window(x, int(starts[i]), length) for i in idx]
```

A `datasplit` on windows gives a train/test partition, but the two sides
interleave in time — the selector picks representative windows throughout
the series, not a chronological cut. `splitquality` on that partition
measures how well the test windows are interpolated from the rest, not
whether a model can forecast ahead of them. For a forecasting evaluation,
fix a chronological holdout first, then select only inside the training
block, with every window built so it lies wholly inside its block.

## Selectors against random

n = 100 of M = 1000 windows (`L = 32`, `p = 3`, so `L*p = 96` columns),
energy distance on the standardized flattened windows. Full table:
[`assets/examples/time_series_windows.md`](assets/examples/time_series_windows.md).

| method | energy distance | regime-proportion error |
|---|---:|---:|
| random | 0.119 ± 0.018 | 0.036 ± 0.031 |
| twinning | 0.0627 | 0.02 |
| herding · energy | 0.0466 | 0 |
| kernel thinning · energy | 0.0483 ± 0.0005 | 0.0067 ± 0.0058 |
| support points · energy (kappa = 300) | 0.121 ± 0.017 | 0.053 ± 0.006 |

The support-point selector did not beat random at 96 columns: its energy
distance and regime-proportion error both sit at the random level. This is
consistent with the rounding rule of `select_nearest`: when the optimizer's
displacement stays below the row spacing, every point rounds back to its
own starting row and the selection is the initial random sample. Prefer
twinning, herding, or kernel thinning for flattened windows.

## Choosing L

`L` has to reach the series' dependence length or the flattened window
carries no more information than a single point. `TwinningSplitter`,
n = 100 of M = 1000 windows, built from only the first `L_short` rows of
each length-32 segment and evaluated in the full `L = 32` space, averaged
over 5 independently generated datasets:

| L_short | ratio to random | regime-proportion error |
|---:|---:|---:|
| 1 | 0.895 | 0.028 |
| 2 | 0.896 | 0.024 |
| 4 | 0.854 | 0.01 |
| 8 | 0.872 | 0.01 |
| 16 | 0.656 | 0.008 |
| 32 | 0.565 | 0.006 |

The advantage over random barely moves until `L_short` passes the series'
dependence length, about 16 here (mean Markov-chain run length
``1/(1-0.94)``). Choose `L` at least as long as the autocorrelation decay
lag, or the seasonal period, of the series you are windowing.

## When windows are long

Flattening trades window length for column count: `L*p` columns per row.
`TwinningSplitter` and `SupportPointSplitter` (kappa = 500) against random,
`M = 2000`, `n = 200`, `p = 3`:

| L·p | method | compile s | run s | ratio to random |
|---:|---|---:|---:|---:|
| 24 | twinning | 0.0001 | 0.017 | 0.378 |
| 24 | support points | 0.16 | 0.16 | 0.918 |
| 192 | twinning | 0.6 | 0.066 | 0.595 |
| 192 | support points | 0.97 | 0.84 | 0.863 |
| 1536 | twinning | 22 | 0.2 | 0.817 |
| 1536 | support points | 33 | 12 | 1.05 |
| 3072 | twinning | 110 | 0.76 | 0.893 |
| 3072 | support points | 44 | 63 | 0.966 |

"Compile seconds" is the first call at that column count — the
width-specific compilation of `TwinningSplitter`'s brute-force search and
`select_nearest`'s k-d tree, both `NearestNeighbors.jl` structures built
over static vectors of the row width. Both selectors lose their edge over
random as `L*p` grows: twinning's ratio rises from 0.38 at 24 columns
to 0.89 at 3,072, and support points reach parity with random by 1,536. At `L*p = 6,144` the first call did not finish within 7 minutes, and
at 12,288 the compiler failed with a memory error. This is a measured limit
of the current release, from the width-specific compilation cost of the
static-vector nearest-neighbor structures, not a design choice.

Past a few hundred columns, do not flatten. Options, roughly in order of
effort:

- **Summary features.** Replace each window by a short vector of
  per-window statistics: mean, variance, lag-k autocorrelations, spectral
  peaks, or a ready-made set such as catch22 (Lubba et al., 2019). Select
  on that vector with `selectrows` as usual.
- **A learned embedding.** Replace each window by a fixed-length embedding
  from an unsupervised time-series representation such as TS2Vec (Yue et
  al., 2022), then select on the embedding. Both of these are
  user-supplied representations, not package dependencies.
- **Rolling origin.** When a window holds thousands of observations and
  consecutive windows overlap or nest (a rolling forecast origin), do not
  flatten at all: the flattened rows would trace a curve through
  ``\mathbb{R}^{Lp}`` rather than filling out a distribution, collapsing
  selection to a time-uniform pick, and `Z` would repeat every observation
  `L/stride` times. Represent each origin by a short state vector — the
  same summary-feature idea, computed up to that origin — and select
  origins with `selectrows` instead of windows.

## Boundaries

- Non-overlapping windows are not statistically independent; only their
  point-level marginals may look exchangeable.
- For a forecasting evaluation, fix the chronological holdout before
  selecting, and select only inside the training block.
- Preprocessing (standardization) must be fit only on data the evaluation
  is allowed to see — the training block, not the full series.
- For series with labeled invalid stretches (a normal-only training set),
  keep the time axis and exclude any window that intersects an invalid
  interval, rather than deleting the invalid points and joining what
  remains: splicing changes the temporal structure the windows are meant
  to capture.
- There is no support here for class quotas across labeled normal and
  anomalous windows; stratified selection by label is out of scope.
- Several windows drawn from one underlying event or recording may need a
  grouping constraint so a selector does not split them apart; that is out
  of scope too.
- Nothing beyond the window length `L`, temporal independence, or
  downstream model performance is guaranteed by any selector here. The
  package selects windows as if they were exchangeable rows; for sampling
  time series under distributional constraints as a problem in its own
  right, see Combes, Fraiman & Ghattas (2022).

## References

- Combes, F., Fraiman, R., & Ghattas, B. (2022). Time Series Sampling. *Engineering Proceedings*, 18(1), 32.
- Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., & Jones, N. S. (2019). catch22: CAnonical Time-series CHaracteristics. *Data Mining and Knowledge Discovery*, 33(6), 1821-1852.
- Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., & Xu, B. (2022). TS2Vec: Towards Universal Representation of Time Series. *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(8), 8980-8987.
