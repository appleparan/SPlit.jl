# splitiq

`splitiq` is a Python wrapper around [SPlit.jl](https://github.com/appleparan/SPlit.jl), a
Julia package for optimal train/test splitting via support points. All computation runs in
Julia through [juliacall](https://github.com/JuliaPy/PythonCall.jl); `splitiq` only converts
data in and out and translates errors, so its results and guarantees are those of SPlit.jl.

## Installation

```bash
pip install splitiq
```

Add the `pandas` extra for DataFrame input:

```bash
pip install "splitiq[pandas]"
```

Python 3.12 or later is required.

## Quick start

```python
import numpy as np
import splitiq

X = np.random.default_rng(1).standard_normal((1_000, 3))
result = splitiq.datasplit(X, ratio=0.2, seed=2)

train, test = result.apply(X)          # or X[result.train_indices], X[result.test_indices]
splitiq.splitquality(X, result)        # energy distance between train and test; lower is better
splitiq.optimal_split_ratio(X[:, :2], X[:, 2])
```

`datasplit` also accepts a pandas DataFrame. `category` columns keep their category order;
plain string/object columns are encoded using their sorted unique values as levels:

```python
import pandas as pd

df = pd.DataFrame({'x': X[:, 0], 'g': pd.Categorical(['a', 'b', 'c'] * (len(X) // 3))})
result = splitiq.datasplit(df, ratio=0.2, seed=2)
train, test = result.apply(df)         # df.iloc[result.train_indices], df.iloc[result.test_indices]
```

## API

| Function | Description |
| --- | --- |
| `datasplit(data, ratio=0.2, *, method, kernel, bandwidth, kappa, max_iterations, tolerance, n_threads, seed)` | Split `data` into train/test sets whose distributions match closely; returns a `SplitResult`. |
| `SplitResult` | `train_indices`, `test_indices` (0-based numpy arrays), `converged`, `iterations`, `method`, `kernel`, `bandwidth`, `ratio`; `.apply(data)` returns `(train, test)`; supports `train_idx, test_idx = result`. |
| `splitquality(data, result, *, kernel, bandwidth, estimator, exact_threshold, seed, n_threads)` | Discrepancy between the train and test rows of `data`; lower is better. |
| `energydistance(x, y, *, estimator, seed, n_threads)` | Energy distance between two samples. |
| `mmd(x, y, kernel='gaussian', *, bandwidth, estimator, seed, n_threads)` | Squared maximum mean discrepancy between two samples. |
| `Exact()`, `Subsample(m, repeats=8)`, `RandomSlices(k)`, `RandomFeatures(D)` | Discrepancy estimators for `energydistance`/`mmd`/`splitquality`. |
| `optimal_split_ratio(x, y, *, method='simple', degree=2)` | Optimal test-set fraction `gamma = 1 / (sqrt(p) + 1)`. |

`method='support_points'` runs the Mak & Joseph (2018) / Joseph & Vakayil (2021) optimizer;
`method='herding'` runs greedy kernel herding. Indices are 0-based. A Julia `ArgumentError`
surfaces as a Python `ValueError`; other Julia errors propagate as `juliacall.JuliaError`.
See the docstrings under `src/splitiq/` for the full argument reference, or build the API
reference locally with `make docs`.

## First call and threads

Julia does not start when you `import splitiq`. It starts on the first call to `datasplit`,
`splitquality`, or any other function that needs it. On that first call, `juliapkg` installs a
compatible Julia (>= 1.10, via `juliaup`) if none is on the `PATH`, instantiates SPlit.jl from
git, and precompiles it. This one-time step takes a few minutes. Later starts (a new process
picking up the already-installed Julia and the precompiled package) take about two seconds.

Julia runs single-threaded inside Python unless `PYTHON_JULIACALL_THREADS` (e.g. `auto`, or a
number) is set in the environment before the first call. The `n_threads` keyword argument only
limits parallelism within the threads Julia was started with; it cannot raise that count.

## Versioning and releases

The `splitiq` version tracks the SPlit.jl version (currently 0.5.0); `src/splitiq/juliapkg.json`
pins SPlit.jl at the git tag `v<version>`. Pushing a `vX.Y.Z` release tag builds the versioned
Julia documentation and, through the `PythonPublish` workflow, publishes `splitiq` to PyPI, so
both releases come from one tag. There is no separate changelog or release script for the
Python package.

## Development

From the `splitiq/` directory:

```bash
uv sync --group dev --group docs   # install dependencies
make julia-dev                     # build .julia_dev/, developing SPlit.jl from this checkout
make test                          # run pytest against .julia_dev/
make format                        # ruff format
make lint                          # ruff check --fix
make typecheck                     # ty check
make docs                          # mkdocs build --strict
make build                         # uv build
```

`make julia-dev` runs `scripts/setup_julia_dev.sh`, which develops SPlit.jl from the repository
checkout instead of the git-pinned revision in `juliapkg.json`, and pins `PythonCall` to the
version `juliacall` itself requires. `make test` runs against that project by setting
`PYTHON_JULIACALL_PROJECT`/`PYTHON_JULIACALL_EXE`.

Pre-commit hooks are configured at the repository root and run from there
(`uvx pre-commit run -a`), not from `splitiq/`.

## References

- Mak, S. & Joseph, V. R. (2018). Support points. *Annals of Statistics*, 46(6A).
- Joseph, V. R. & Vakayil, A. (2021). SPlit: An optimal method for data splitting.
  *Technometrics*, 63(4).
- Joseph, V. R. (2022). Optimal ratio for data splitting. *Statistical Analysis and Data
  Mining*, 15(4).
- Chen, Y., Welling, M. & Smola, A. (2010). Super-samples from kernel herding. *UAI*.

This project template is generated by [copier-modern-ml](https://github.com/appleparan/copier-modern-ml).
