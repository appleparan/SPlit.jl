# Overview

`splitiq` is a Python wrapper around [SPlit.jl](https://github.com/appleparan/SPlit.jl). It
carries no algorithm code of its own: every call converts its arguments to Julia values, invokes
the corresponding SPlit.jl function through [juliacall](https://github.com/JuliaPy/PythonCall.jl),
and converts the result back. The optimizer, the energy-distance and MMD computations, and the
kernel-herding selection all run inside the embedded Julia process, not in Python.

## Type mapping

| Python | Julia |
| --- | --- |
| 1-D or 2-D numpy array | `Vector`/`Matrix` (reshaped to a single column if 1-D) |
| pandas DataFrame | `DataFrame`, built column by column |
| numeric or boolean DataFrame column | `Vector{Float64}` |
| `category` DataFrame column | `CategoricalVector`, category order preserved |
| string/object DataFrame column | `CategoricalVector`, levels sorted |
| `seed: int` | `Random.Xoshiro(seed)` |
| `Exact()`, `Subsample`, `RandomSlices`, `RandomFeatures` | the SPlit.jl estimators of the same name |
| `'energy'` / `'gaussian'` | `EnergyKernel()` / `GaussianKernel(...)` |
| Julia `ArgumentError` | Python `ValueError` |
| any other Julia exception | `juliacall.JuliaError` |
| 0-based numpy index array | 1-based Julia row index vector |
| `'farthest'` / `'random'` / 0-based `start` | `:farthest` / `:random` / 1-based row index |
| `'auto'` / `'always'` / `'never'` `compress` | `:auto` / `:always` / `:never` |
| `standardize: bool` | `standardize::Bool` (`False` skips preprocessing entirely) |

## `SplitResult`

`datasplit` returns a frozen `SplitResult` dataclass:

- `train_indices`, `test_indices`: 0-based numpy `int64` arrays partitioning the input rows.
- `converged`: whether the optimizer's stopping rule fired (always `True` for kernel herding
  and kernel thinning, which have no iterative convergence criterion).
- `iterations`: number of optimizer iterations (kernel herding: number of greedy selections;
  kernel thinning: number of KT-SWAP replacements).
- `method`: `'support_points'`, `'herding'`, `'twinning'`, or `'kernel_thinning'`.
- `kernel`: `'energy'` or `'gaussian'`.
- `bandwidth`: the resolved Gaussian bandwidth, or `None` for the energy kernel.
- `ratio`: the requested test-set fraction.

`result.apply(data)` returns `(train, test)` subsets of `data` (numpy fancy indexing, or
`.iloc` for a pandas DataFrame/Series), and `train_idx, test_idx = result` unpacks the two
index arrays directly.

## `SplitComparison`

`compare` returns a frozen `SplitComparison` dataclass: `results` (one `SplitResult` per
entry of the `methods` argument) and `qualities` (one discrepancy score per result, under
`kernel`; lower is better), both index-aligned with `methods`, plus the scoring `kernel`
itself (`'energy'` or `'gaussian'`) — distinct from any per-method `kernel` a `methods`
mapping entry sets for its own splitter. `comparison.best()` returns the `(index, result)`
pair with the lowest quality.

## Versioning

The `splitiq` version tracks the SPlit.jl version; `src/splitiq/juliapkg.json` pins SPlit.jl at
the git tag `v<version>`. Pushing a `vX.Y.Z` release tag builds the versioned Julia
documentation and, through the `PythonPublish` workflow, publishes `splitiq` to PyPI, so both
releases come from one tag.

## Method details

For the splitting methods, kernels, and quality diagnostics themselves, the properties they
guarantee, and the papers they come from, see the
[Julia documentation](https://liam.kim/SPlit.jl). `splitiq` changes none of that
behavior; it only exposes it to Python.
