# Phase 2b — Kernel Herding Splitter and Benchmarks

**Status**: Approved design, pre-implementation
**Date**: 2026-09-02
**Branch**: `feat/kernel-herding`
**Builds on**: Phase 1 (`2026-09-02-paper-aligned-redesign-design.md`) and Phase 2a (`2026-09-02-gaussian-kernel-design.md`)

## TL;DR

Add `HerdingSplitter`, a second splitter that selects data rows directly by
greedy kernel herding (Chen, Welling & Smola 2010) instead of optimizing
continuous support points and mapping them back. It works with both
`GaussianKernel` and `EnergyKernel` (for the latter it is greedy
energy-distance minimization). To admit a second splitter, the result and
comparison types are generalized over an `AbstractSplitter` supertype without
changing any public constructor. A reproducible benchmark script compares
all splitting methods on synthetic data, and its results table lives in the
documentation. Non-breaking, v0.4.0.

## Source of truth

- Chen, Y., Welling, M., & Smola, A. (2010). Super-Samples from Kernel
  Herding. *UAI 2010* (arXiv:1203.3472). — greedy rule Eq. (8), error
  Eq. (9), $O(1/T)$ rate (Proposition 1), approximate maximization
  (Corollary 2), herding on an empirical distribution with the argmax taken
  over the data points (Sections 3.1.2 and 3.2).
- Gretton et al. (2012) for the MMD; Mak & Joseph (2018) / Joseph & Vakayil
  (2021) for the support-point path, which is untouched.

## The rule and why it is greedy MMD minimization

With data rows $x_1, \dots, x_N$ (after `preprocess`), kernel $k$, and the
rows selected so far $s_1, \dots, s_T$, the paper's Eq. (8) on the empirical
distribution reads

```math
s_{T+1} = \arg\max_{x \in \{x_1,\dots,x_N\} \setminus \{s_1,\dots,s_T\}}
  \; \frac{1}{N} \sum_{l=1}^{N} k(x, x_l) \;-\; \frac{1}{T+1} \sum_{t=1}^{T} k(x, s_t).
```

Restricting the argmax to rows not yet selected is this package's one
deviation from the paper (which allows repeats): a split needs a subset.

The V-statistic $\mathrm{MMD}^2$ between the selected set and the data
(Phase 2a) changes, when $x$ is appended to $T$ selected rows, by

```math
\Delta(x) = \frac{k(x,x)}{(T+1)^2} + \frac{2}{(T+1)^2} \sum_{t=1}^{T} k(x, s_t)
  - \frac{2}{(T+1)N} \sum_{l=1}^{N} k(x, x_l) + \text{const},
```

so for kernels with constant $k(x,x)$ — Gaussian ($=1$) and energy
($k(u,v) = -\|u-v\|$, $=0$) — the row maximizing Eq. (8) is exactly the row
minimizing $\Delta$: each herding step is the greedy step of MMD² (energy
distance) minimization. The first selection maximizes the data term alone and
is deterministic; the whole procedure is deterministic given the kernel and
the data (ties broken by the lowest row index).

Proposition 1 gives $O(1/T)$ decrease of Eq. (9) when $\|\phi(x)\|$ is
bounded (true for the Gaussian kernel); Corollary 2 keeps the rate under
approximate maximization, which justifies estimating the data term from a
row subsample (`kappa`).

## Architecture

### Splitter hierarchy (`src/splitter.jl`)

```julia
abstract type AbstractSplitter end
struct SupportPointSplitter{K,R} <: AbstractSplitter   # unchanged fields
struct SplitResult{M<:AbstractSplitter}                # method::M; same positional ctor
```

`SplitComparison.methods` becomes `Vector{AbstractSplitter}`; `compare`
accepts `Vector{<:AbstractSplitter}`. `datasplit`, `train_indices`,
`test_indices`, iteration and `getindex` sugar are unchanged.

### `HerdingSplitter` (`src/herding.jl`)

```julia
HerdingSplitter(; kernel = GaussianKernel(), ratio = 0.2, kappa = nothing,
                  n_threads = Threads.nthreads(), rng = Random.default_rng())
datasplit(::HerdingSplitter, data) -> SplitResult
```

- `preprocess` → `resolve(kernel, X, rng)` → `herd(kernel, X, n_small; kappa,
  n_threads, rng)` → the selected rows are the smaller subset (test when
  `ratio ≤ 0.5`, train otherwise), exactly as for support points.
- `SplitResult.converged = true`, `iterations = n_small` (one selection per
  iteration; the procedure always terminates).
- `herd` computes the data term $d_i = \frac{1}{N}\sum_l k(x_i, x_l)$ for
  every row once — $O(N^2)$, chunked over `n_threads` with block-wise
  accumulation, no $N\times N$ matrix — or, with `kappa`, from `kappa` rows
  drawn with `rng` (unbiased estimate, Corollary 2). It then keeps a running
  sum $c_i = \sum_t k(x_i, s_t)$ updated in $O(N)$ per selection. Total
  $O(N^2 + nN)$ time, $O(N)$ extra memory.
- `EnergyKernel` gains `kernelvalue(k, u, v) = -\|u - v\|` so both kernels
  share the herding code; the energy kernel's `k(x,x)=0` keeps the greedy
  equivalence above.

### Quality and comparison

Unchanged in behavior: `splitquality(data, result; kernel)` and
`compare(methods, data; kernel, rng)` work for any `AbstractSplitter`.
`DataFrame(comparison)` gains a `method` column (`"SupportPointSplitter"` /
`"HerdingSplitter"`) before `kernel`.

### Benchmarks (`benchmark/`)

- `benchmark/Project.toml` (SPlit as a dev dependency, DataFrames, Random,
  Statistics — no BenchmarkTools; `@elapsed` after one warm-up run) and
  `benchmark/run.jl`.
- Datasets (seeded): 2-D Gaussian mixture (4 components), 10-D standard
  normal, 5-D uniform, 3-D heavy-tailed ($t_3$); $N \in \{1{,}000, 10{,}000\}$;
  `ratio = 0.2`.
- Methods: `SupportPointSplitter(EnergyKernel())`,
  `SupportPointSplitter(GaussianKernel())`, `HerdingSplitter(EnergyKernel())`,
  `HerdingSplitter(GaussianKernel())`, random split (mean of 5 seeds).
- Metrics per (dataset, method): energy distance and Gaussian-kernel MMD of
  the split (median-heuristic bandwidth resolved once per dataset), wall
  time; support-point methods use `kappa = 1_000` at $N = 10{,}000$.
- Output: a Markdown table on stdout plus figures written with CairoMakie
  (a dependency of the benchmark environment only, never of the package) to
  `docs/src/assets/benchmarks/`:
  1. `quality.png` — grouped bars of energy distance and Gaussian MMD per
     method, one panel per dataset (log scale, random split as the
     reference);
  2. `time.png` — wall time versus $N$ per method (log–log);
  3. `selection.png` — the 2-D Gaussian-mixture data with the test rows
     chosen by each method overlaid, one panel per method (the visual
     argument for why the methods differ).
  `docs/src/20-benchmarks.md` embeds the three figures and the committed
  table, and records the exact command, Julia/thread/CPU info, and a short
  reading of the results (which method to prefer when). Re-running the script
  regenerates the table and the figures; both are committed so the docs
  build needs no benchmark run.

## Documentation (deliverable)

- `docs/src/10-methods.md`: new "Kernel herding" section — Eq. (8) on the
  empirical distribution, the $\Delta$ derivation above, the $O(1/T)$
  statement, the `kappa` estimate, and the function names (`herd`,
  `HerdingSplitter`).
- `docs/src/20-benchmarks.md`: as above, with the three figures embedded
  (`![…](assets/benchmarks/….png)`; Documenter copies `docs/src/assets`).
- `docs/src/index.md`: a Splitters paragraph showing both splitters;
  README: one bullet and one snippet for `HerdingSplitter`; AGENTS.md: one
  gotcha (herding is deterministic given the kernel; `rng` only feeds `kappa`
  and `:median`).
- Docstrings with examples for `HerdingSplitter` and `herd`.

## Testing (paper-property style)

1. Greedy correctness: on a small dataset, each herding selection equals the
   brute-force $\arg\min_x \Delta(x)$ over unselected rows (both kernels).
2. MMD² (energy distance for `EnergyKernel`) between the selected set and the
   data is non-increasing along the selection sequence beyond the first step.
3. Optimality: herding splits beat random splits under both `mmd` and
   `energydistance` (both kernels).
4. Determinism: same data and numeric kernel ⇒ identical indices regardless
   of `rng` and `n_threads`; with `kappa`, same `rng` ⇒ identical indices.
5. `compare` with a mixed vector of `SupportPointSplitter` and
   `HerdingSplitter`; `DataFrame` shows the `method` column; `best` works.
6. Validation: `ratio`, `kappa`, unresolved kernel, unsupported kernel type
   errors; `DataFrame`/vector inputs.
7. Benchmark script smoke test: runs on tiny sizes (`--quick` flag), prints
   a table with the expected rows, and writes the three PNG files to a
   temporary output directory (`--out`).

## Non-goals

- Weighted herding / Frank–Wolfe line-search variants.
- Herding under other kernels (needs `kernelvalue` only — add on demand).
- Parallelizing the per-selection $O(N)$ update.

## Breaking changes

None. New exports: `AbstractSplitter`, `HerdingSplitter`. `SplitResult`'s
type parameters change from `{K,R}` to `{M}` (positional constructor and
field names unchanged). Version 0.3.0 → 0.4.0.
