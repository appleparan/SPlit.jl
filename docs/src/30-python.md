# Python

The Python package [splitiq](https://pypi.org/project/splitiq/) exposes
SPlit.jl to Python through
[juliacall](https://github.com/JuliaPy/PythonCall.jl). It is a thin wrapper:
every computation runs in Julia, so the properties documented on the
[Methods](10-methods.md) page hold unchanged.

```bash
uv add splitiq        # or, without uv: pip install splitiq
```

Importing `splitiq` does not start Julia. The first call to `datasplit`, or
any other function, starts it: `juliapkg`/`juliaup` install a compatible
Julia if none is on the `PATH`, and SPlit.jl is precompiled. That one-time
step takes a few minutes; later starts take about two seconds.

```python
import numpy as np
import splitiq

X = np.random.default_rng(1).standard_normal((1_000, 3))
result = splitiq.datasplit(X, ratio=0.2, seed=2)

train, test = result.apply(X)          # or X[result.train_indices]
splitiq.splitquality(X, result)        # energy distance, lower is better
splitiq.optimal_split_ratio(X[:, :2], X[:, 2])
```

`datasplit` accepts numpy arrays and pandas DataFrames. String and
`category` columns are Helmert-encoded exactly as a Julia `DataFrame` with a
`CategoricalVector` would be. Indices are 0-based numpy arrays.

| Python | Julia |
|---|---|
| `datasplit(X, ratio, method="support_points", kernel="energy", kappa=..., seed=...)` | `datasplit(SupportPointSplitter(...), X)` |
| `datasplit(X, ratio, method="herding", kernel="gaussian", bandwidth=...)` | `datasplit(HerdingSplitter(...), X)` |
| `datasplit(X, ratio, method="twinning", ...)` | `datasplit(TwinningSplitter(...), X)` |
| `datasplit(X, ratio, method="kernel_thinning", ...)` | `datasplit(KernelThinningSplitter(...), X)` |
| `select_rows(X, n, ...)` | `selectrows(splitter, X, n)` |
| `multiplet(X, k, strategy=..., ...)` | `multiplet(splitter, X, k; strategy)` |
| `compare(X, methods)` / `SplitComparison.best()` | `compare(methods, X)` / `best` |
| `splitquality`, `energydistance`, `mmd` with `Exact`, `Subsample`, `RandomSlices`, `RandomFeatures` | same names and estimators |
| `optimal_split_ratio(x, y)` | `optimal_split_ratio(x, y)` |
| `seed=<int>` | `rng = Xoshiro(seed)` |
| 0-based numpy arrays of indices | 1-based `Vector{Int}` of indices |

Julia runs single-threaded inside Python unless `PYTHON_JULIACALL_THREADS`
is set (for example to `auto`) before the first call.

The Python package shares its version number with SPlit.jl. Release `vX.Y.Z`
of `splitiq` pins SPlit.jl at the git tag `vX.Y.Z`, and pushing that tag
publishes the Python package to PyPI, so both releases come from one tag. The package sources live
in the `splitiq/` directory of the repository; its own documentation covers
installation details and the development setup.

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
samples. This mirrors the Julia `weights`, `weights_x`, and `weights_y`
keywords described in [Methods](@ref weighted-samples).

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
combined with `reference`. This mirrors the Julia `reference` and
`reference_weights` keywords described in
[Methods](@ref reference-distribution).

## Twinning and k-fold multiplets

`method='twinning'` selects `TwinningSplitter`; `start` picks the starting row
(`'farthest'`, `'random'`, or a 0-based index). `multiplet` returns `k` folds:

```python
import numpy as np
from splitiq import datasplit, multiplet

data = np.random.default_rng(0).standard_normal((100_000, 10))
result = datasplit(data, ratio=0.2, method='twinning')
folds = multiplet(data, 5)                        # method='twinning', strategy='sequential'
```

See [Methods](@ref twinning) for the algorithm and the three strategies.

## Kernel thinning

`method='kernel_thinning'` selects `KernelThinningSplitter`; `delta` sets
the failure probability of the kernel-thinning guarantees and is only
valid with this method (default `0.5`):

```python
import numpy as np
from splitiq import datasplit

data = np.random.default_rng(0).standard_normal((10_000, 5))
result = datasplit(data, ratio=0.2, method='kernel_thinning', kernel='energy', delta=0.5, seed=7)
```

`kernel='gaussian'` (with `bandwidth`) and `n_threads` apply as with the
other methods; `kappa`, `max_iterations`, `tolerance`, and `start` raise
`ValueError` with this method, as they do with `'herding'`. See
[Kernel thinning](@ref kernel-thinning) for the algorithm.

## Embeddings

`standardize=False` uses a numeric array exactly as given, with no
Helmert encoding, constant-column removal, or per-column scaling — the
mode for cosine-normalized embeddings, where standardizing columns would
distort the angles. It is accepted by `datasplit`, `select_rows`,
`multiplet`, `splitquality`, and `compare`; a pandas DataFrame raises
`ValueError` with it, because a DataFrame needs the encoding step.

`compress` (kernel thinning only) picks the Compress++ path: `'auto'`
(the default) uses it when `n` is a small fraction of the row count and
the target measure is the data itself, `'always'` and `'never'` force
either path, and `'always'` with `weights` or `reference` raises
`ValueError`.

```python
import numpy as np
from splitiq import select_rows, energydistance

E = np.random.default_rng(0).standard_normal((5_000, 384))
E /= np.linalg.norm(E, axis=1, keepdims=True)

idx = select_rows(E, 500, method='herding', kernel='energy', standardize=False)
few = select_rows(E, 100, method='kernel_thinning', standardize=False)   # compress='auto'
energydistance(E[idx], E)
```

The "Skipping preprocessing" section of [Methods](@ref methods) and
[Compress++](@ref compress) describe what each does;
[Selecting LLM training data](@ref llm-data-selection) has the workflow and
the decision table.
