# Getting Started

## Installation

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv add splitiq
```

Add the `pandas` extra for DataFrame input:

```bash
uv add "splitiq[pandas]"
```

If you are not using uv, `pip` works as well:

```bash
pip install splitiq            # or: pip install "splitiq[pandas]"
```

Python 3.12 or later is required. No separate Julia installation is required beforehand: the
first call installs one if needed (see [First call](#first-call) below).

## First call

Julia does not start when you `import splitiq`. It starts on the first call to `datasplit`,
`splitquality`, `energydistance`, `mmd`, or `optimal_split_ratio`. On that first call, `juliapkg`
installs a compatible Julia (>= 1.10, via `juliaup`) if none is already on the `PATH`,
instantiates SPlit.jl from git, and precompiles it. This one-time step takes a few minutes.
Later starts, once Julia and the precompiled package are in place, take about two seconds.

```python
import splitiq

# The first call in a process pays the one-time setup cost.
splitiq.optimal_split_ratio([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
```

## Basic usage

```python
import numpy as np
import splitiq

X = np.random.default_rng(1).standard_normal((1_000, 3))
result = splitiq.datasplit(X, ratio=0.2, seed=2)

train, test = result.apply(X)          # or X[result.train_indices], X[result.test_indices]
```

`result` is a `SplitResult`: `train_indices` and `test_indices` are 0-based numpy arrays that
partition the rows of `X`; unpacking `train_idx, test_idx = result` also works.

### pandas input

```python
import pandas as pd

df = pd.DataFrame({'x': X[:, 0], 'g': pd.Categorical(['a', 'b', 'c'] * (len(X) // 3))})
result = splitiq.datasplit(df, ratio=0.2, seed=2)
train, test = result.apply(df)         # df.iloc[result.train_indices], df.iloc[result.test_indices]
```

Numeric and boolean columns become float columns. `category` columns keep their category
order; plain string/object columns are encoded using their sorted unique values as levels.
Missing values raise `ValueError`.

## Choosing a ratio

`optimal_split_ratio` computes the fraction Joseph (2022) recommends for the test set, from the
number of predictor columns:

```python
p_predictors = X[:, :2]
response = X[:, 2]
ratio = splitiq.optimal_split_ratio(p_predictors, response)
result = splitiq.datasplit(X, ratio=ratio, seed=2)
```

## Checking split quality

`splitquality` reports the discrepancy between the train and test rows of a completed split;
lower is better, and it is comparable across splits computed on the same data:

```python
quality = splitiq.splitquality(X, result)
```

By default it computes exactly below 20,000 total rows and switches to an estimator above that
threshold (`exact_threshold`); pass `estimator=` explicitly to control that yourself.

## Options

- `method='support_points'` (default) runs the Mak & Joseph (2018) / Joseph & Vakayil (2021)
  majorization-minimization optimizer; `method='herding'` runs greedy kernel herding instead,
  which is deterministic given the data and a numeric kernel and has no `kappa`,
  `max_iterations`, or `tolerance` options.
- `kernel='energy'` (default) or `kernel='gaussian'`; the Gaussian kernel's bandwidth defaults
  to `'median'`, resolved from the data, or accepts a fixed positive number.
- `kappa` bounds the number of rows the optimizer considers per iteration. Set it below the
  number of rows to switch to stochastic majorization-minimization on large datasets;
  `method='support_points'` only.
- `seed` fixes the RNG for reproducible splits (`Random.Xoshiro(seed)` on the Julia side).
  Without a seed, Julia's default RNG is used.

## Troubleshooting

**The first call is slow.** This is expected the first time a compatible Julia and a
precompiled SPlit.jl are installed on a machine; it takes a few minutes. Subsequent calls,
including in new processes, reuse the installed Julia and precompiled package and take about
two seconds.

**Julia only uses one thread.** Set `PYTHON_JULIACALL_THREADS` (e.g. to `auto`, or a specific
number) in the environment *before* the first call to `splitiq`. It is read when Julia starts
and cannot be changed afterward in the same process. The `n_threads` argument on `datasplit`,
`splitquality`, `energydistance`, and `mmd` only limits parallelism within the threads Julia
was started with.

**Controlling where Julia and its packages are installed.** Set `JULIA_DEPOT_PATH` before the
first call to redirect where `juliaup`/`juliapkg` place the Julia installation and package
depot.

**Offline or restricted machines.** `juliapkg`'s automatic Julia installation needs network
access. On an offline machine, install a compatible Julia (>= 1.10) yourself and make sure it
is on the `PATH` before the first call; `juliapkg` will use it instead of downloading one.

## Development setup

From the `splitiq/` directory:

```bash
uv sync --group dev --group docs   # install dependencies
make julia-dev                     # build .julia_dev/, developing SPlit.jl from this checkout
make test                          # run pytest against .julia_dev/
```

`make julia-dev` runs `scripts/setup_julia_dev.sh`, which develops SPlit.jl from the repository
checkout instead of the git-pinned revision in `src/splitiq/juliapkg.json`. Run it once before
`make test`. See the [README](index.md#development)
for the full set of `make` targets.

## Weighted samples

Pass `weights` (one non-negative entry per row) to make the split target
the weighted distribution of the rows, for example a quality score per
sample:

```python
import numpy as np
from splitiq import datasplit, splitquality

data = np.random.default_rng(0).standard_normal((1000, 8))
weights = np.exp(np.random.default_rng(1).standard_normal(1000))

result = datasplit(data, ratio=0.2, seed=42, weights=weights)
print(splitquality(data, result, weights=weights))
```

`energydistance` and `mmd` take `weights_x` and `weights_y` for their two
samples. Weights proportional to duplication counts are equivalent to
duplicating rows.

## Selecting rows toward a reference

`select_rows` returns the indices of `n` rows without forming a partition, and
`reference` makes the chosen rows follow a target sample instead of the
data itself:

```python
import numpy as np
from splitiq import select_rows, splitquality, datasplit

data = np.random.default_rng(0).standard_normal((5000, 16))
target = np.random.default_rng(1).standard_normal((800, 16)) + 0.5

idx = select_rows(data, 500, seed=42, reference=target)          # 0-based row indices
result = datasplit(data, ratio=0.1, seed=42, reference=target)
print(result.selected, splitquality(data, result, reference=target))
```

`reference_weights` weights the reference rows; `weights` cannot be
combined with `reference`.
