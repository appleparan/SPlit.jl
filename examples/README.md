# Examples

`llm_data_selection.jl` selects LLM training data from a public arXiv-abstract
embedding matrix with every `SPlit` splitter, under plain/weighted/targeted
settings, and times Compress++ against plain kernel thinning.

Run it from the repository root (`path="."` is the checkout — SPlit is a
registered package, so a plain `Pkg.instantiate()` resolves the registry
version instead):

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia -t auto --project=examples examples/llm_data_selection.jl
```

The resulting table is written to `docs/src/assets/examples/llm_selection.md`.

`time_series_windows.jl` flattens fixed-length time-series windows into rows
(`L` timesteps × `p` variables → one row of length `L*p`, variable-major) so
a splitter can select a distribution-preserving subset of windows. It walks
through the fixture from the design doc, checks that point-level moments
match across two regimes with the same mean/variance but different temporal
dependence while window-level distributions differ, compares every splitter
against random, demonstrates recovering the original time-series slice for
selected windows, and runs two contrasts: representing less than the
dependence length, and representing far more dimensions than a splitter can
use well (the `L*p` ladder, which can take minutes at the largest `L`).

Run it the same way, from the repository root:

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia -t auto --project=examples examples/time_series_windows.jl
```

Use `--quick` for a fast smoke run (small sizes, a short `L` ladder) and
`--out PATH` to change the output file. The resulting tables are written to
`docs/src/assets/examples/time_series_windows.md`.

## Python

Both examples have a Python counterpart under `splitiq/examples/`, run from
`splitiq/`:

```sh
uv run python examples/llm_data_selection.py
uv run python examples/time_series_windows.py
```
