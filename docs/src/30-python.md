# Python

The Python package `splitiq` exposes SPlit.jl to Python through
[juliacall](https://github.com/JuliaPy/PythonCall.jl). It is a thin wrapper:
every computation runs in Julia, so the properties documented on the
[Methods](10-methods.md) page hold unchanged.

```bash
uv add splitiq        # or, without uv: pip install splitiq
```

The first `import splitiq` call installs Julia (through juliaup, if no
compatible Julia is on the path) and precompiles SPlit.jl. That one-time step
takes a few minutes; later imports start in a couple of seconds.

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
| `splitquality`, `energydistance`, `mmd` with `Exact`, `Subsample`, `RandomSlices`, `RandomFeatures` | same names and estimators |
| `optimal_split_ratio(x, y)` | `optimal_split_ratio(x, y)` |
| `seed=<int>` | `rng = Xoshiro(seed)` |

Julia runs single-threaded inside Python unless `PYTHON_JULIACALL_THREADS`
is set (for example to `auto`) before the first import.

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
