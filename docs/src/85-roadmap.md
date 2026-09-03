# [Roadmap](@id roadmap)

This page records where SPlit.jl is, where it is going, and why. It is a
living document: milestones move between sections as work progresses, and
open questions are resolved in place with a short note.

## Vision

SPlit.jl started as a Julia implementation of SPlit (Joseph & Vakayil,
2022, *Technometrics* 64(2)): an optimal train/test splitting method for
tabular data based on support points. The literature that followed SPlit
has largely reframed the problem as distribution compression, choosing a
subset whose empirical distribution stays close to a reference distribution
under energy distance or MMD.

The roadmap moves SPlit.jl from a data-splitting package toward a
distribution-preserving subset selection library that serves two
audiences with one interface:

1. Tabular data science: optimal train/test and k-fold splits, the
   original use case.
2. Embedding-based data selection: choosing training subsets for LLM
   fine-tuning or pretraining from embedding matrices (hundreds to
   thousands of dimensions, N in the hundreds of thousands), where the
   reference distribution may be a target set rather than the data itself,
   and samples may carry quality weights.

## Current state

State of the exported API at v0.5.2.

| Component | Status | Notes |
|---|---|---|
| `SupportPointSplitter` with `EnergyKernel` | done | MM update minimizing energy distance (Mak & Joseph, 2018); `kappa` gives the stochastic subsampled variant of Joseph & Vakayil (2022); `select_nearest` rounds optimized points to data rows via a k-d tree. |
| `SupportPointSplitter` with `GaussianKernel` | done | Minimizes squared MMD by projected gradient descent with Armijo backtracking. Has no `kappa` mode: the `SupportPointSplitter` constructor throws for `GaussianKernel` with `kappa` set. A `:median` bandwidth is resolved at `datasplit` time and the resolved kernel is stored in `result.method.kernel`. |
| `HerdingSplitter` | done | Greedy kernel herding (Chen, Welling & Smola, 2010); exact `O(N^2)` data term, deterministic given the data and a numeric kernel. |
| `optimal_split_ratio` | done | γ = 1/(√p + 1) (Joseph, 2022). |
| Preprocessing | done | Helmert encoding of categorical columns in canonical level order, constant-column removal, standardization to mean 0 and variance 1. |
| Quality diagnostics | done | `energydistance`, `mmd`, `splitquality` with `Exact`, `Subsample`, `RandomSlices`, `RandomFeatures` estimators and automatic fallback above `exact_threshold`; see [Design experiments](@ref design-experiments). |
| `compare` / `best` | done | Splitter comparison on one dataset. |
| Python package `splitiq` | done | Wraps the Julia package through juliacall; every computation still runs in Julia. See [Python](30-python.md). |
| Weighted samples | done | `weights` on `datasplit`, `splitquality`, `compare`; `weights_x`/`weights_y` on `energydistance` and `mmd`; see [Methods](@ref weighted-samples). |
| Reference (target) distribution | done | `reference`/`reference_weights` on `selectrows`, `datasplit`, `splitquality`, `compare`; see [Methods](@ref reference-distribution). |
| `selectrows` (selection without a partition) | done | Returns the chosen row indices; `datasplit` builds on it. |
| `TwinningSplitter` | done | Sequential nearest-neighbor twinning (Vakayil & Joseph, 2022); energy distance objective, no kernel or optimizer options; deterministic with `start = :farthest`. |
| k-fold splitting (`multiplet`) | done | Strategies S1/S2/S3 of the twinning paper; S1/S2 work with every splitter. |
| High-dimensional data (p in the hundreds) | partly measured | Twinning measured at p = 768 (N = 10⁴) on the [Design experiments](@ref twinning-trees) page; the search structure switches by dimension. Support-point and herding splitters remain untested above p = 10. |

## Design principles

- Backward compatibility. Existing public functions keep their signatures
  and numerical results. New capability is added via new types, new
  functions, or keyword arguments with defaults that reproduce current
  behavior.
- One interface, many algorithms. Every selection method is an
  `AbstractSplitter` with a `datasplit` method: input data (matrix,
  `DataFrame`, or vector, observations in rows), output a `SplitResult`.
  A new method is a new `AbstractSplitter` subtype with `_select_rows` and
  `_with_kernel` methods; `datasplit` and `selectrows` are generic over
  `AbstractSplitter`, not changed for each new one.
- Faithful to the source. Each algorithm follows its original paper.
  Deviations are documented in the docstring under a "Differences from the
  paper" heading.
- Pure Julia core. The Julia package takes no Python or GPU dependency;
  `splitiq` remains a thin wrapper over it, not the other way around.
- Measurable. Every method is tested against the criterion that energy
  distance or MMD to the reference is smaller than a uniform random subset
  of the same size.
- Paper-defined correctness. Correctness is judged against the source
  papers, not against any other implementation. Tests encode the properties those papers guarantee
  (monotone descent, beating random splits, reproducibility under a fixed
  `rng`), and all randomness flows through the caller's `rng`.

## Milestones

Ordered by dependency. Each milestone is a self-contained PR or small PR
series.

### M1: weighted samples

Done (2026-09-03). Add `weights::AbstractVector` to `energydistance`, `mmd`,
and the `support_points` optimizer, plus `HerdingSplitter`'s data term.

- Empirical distribution terms become weighted averages; formulas go in
  docstrings.
- The MM update and the MMD gradient must be re-derived with weights;
  keep the derivation as comments.
- The `DiscrepancyEstimator` methods (`Subsample`, `RandomSlices`,
  `RandomFeatures`) also need weighted forms, added as new methods, never
  as `if` branches, matching how estimator/kernel combinations are already
  organized.
- How `kappa` subsampling interacts with weights was decided by
  experiment: see [Design experiments](@ref weighted-kappa).
- Tests: uniform weights reproduce current results exactly; concentrated
  weights pull support points toward the weighted cluster.

Why first: the smallest change, and M2-M5 build on it. Weighting is also
the piece missing from the compression literature (Twinning, Kernel
Thinning), and the piece LLM data-selection pipelines need to combine
distribution matching with quality scores.

### M2: reference (target) distribution

Done (2026-09-03). Add a `reference::AbstractMatrix` keyword.
Distances are computed against `reference` instead of the data; selection
still happens among the data's own rows.

- `preprocess` must become fit/apply, so the same transform can be fit on
  one set and applied to both.
- Tests: passing a sub-population of the data as reference concentrates
  selection in that sub-population.

Why: turns SPlit into a target-matching selector with a distance-based
rather than density-ratio-based criterion.

`selectrows(splitter, data, n; reference, reference_weights)` ships
alongside, as the selection-only entry point the embedding workflow (M5)
needs.

### M3: twinning and k-fold multiplets

Done (2026-09-03): added
`TwinningSplitter <: AbstractSplitter` implementing the sequential
kd-tree assignment of Vakayil & Joseph (2022), and a `multiplet` function
returning k distribution-balanced folds.

- Benchmarks at N in {10^4, 10^5, 10^6}, p = 10 against the current
  splitters (time and energy distance), plus a p = 768 case to quantify
  k-d tree degradation. Scripts live in `benchmark/`, results on the
  [Benchmarks](@ref benchmarks) page, matching the existing pattern.

Why: Twinning is the direct successor to SPlit, orders of magnitude
faster than the support-point splitters, and gives k-fold splitting for
free.

Twinning takes no `weights`/`reference` (not defined by the paper);
`multiplet(:sequential/:halving)` forwards them to the other splitters.

### M4: kernel thinning backend

Planned, depends on M1 for weighted MMD. Add
`KernelThinningSplitter <: AbstractSplitter` (KT-SPLIT + KT-SWAP, Dwivedi
& Mackey, 2024), with Compress++ (Shetty, Dwivedi & Mackey, 2022) as an
optional wrapper. Reuses the existing `GaussianKernel` type.

- Output size is a power of two; document how arbitrary `n` is handled.
- Tests: on a Gaussian mixture, MMD is significantly below a uniform
  random subset.

Why: selects directly from the data, with no continuous optimization and
no nearest-neighbor assignment step, and comes with an MMD rate of
O(sqrt(log n / n)) that neither the support-point splitters nor Twinning
provide. Near-linear time makes it the realistic option at LLM scale.

### M5: embedding workflow, docs, and comparison

Planned, depends on M1-M4.

- An example script under `examples/`: load an embedding matrix,
  cosine-normalize, select with the M1-M4 combinations, compare energy
  distance against uniform random and K-center greedy.
- A new docs page for selecting LLM training data: a decision table (by
  N, p, weighted?, target?) for which method to use.
- Extend [Methods](@ref methods) with the new methods in the existing
  format.

### M6: MMD gradient-flow update (exploratory)

Idea. Replace the Armijo projected gradient in the Gaussian-kernel path
with the mean-shift-style update from MMD gradient-flow quantization
(arXiv 2502.10600). Structurally similar to the current MM step. The
Gaussian path has no `kappa` stochastic mode today, unlike `EnergyKernel`,
so a cheaper update rule matters more there. Evaluate only after M4 gives
a baseline.

## Open questions

- Weighted `kappa` subsampling (M1). Resolved 2026-09-03 by the experiment
  on the [Design experiments](@ref weighted-kappa) page.
- High-dimensional nearest neighbours (M3). A k-d tree is the wrong
  structure for p around 768. Options: brute-force with BLAS, a
  NearestNeighbors.jl ball tree, or random projection before assignment.
  Benchmark before choosing. Resolved 2026-09-03 by the measurement on the
  [Design experiments](@ref twinning-trees) page: brute force from
  p = 50 (`TWINNING_BRUTE_FORCE_DIMENSION = 50`).
- Categorical handling in embedding mode (M5). Helmert contrasts do not
  apply. Should embedding mode bypass preprocessing entirely, or expose a
  separate preprocessing entry point?
- Is weighted energy distance the right combination rule? Combining a
  quality score with distribution matching via a weighted empirical
  distribution is natural but not validated in the literature. M5's
  comparison is the first test; if it underperforms, alternatives include
  stratified selection by quality quantile.

## References

1. Joseph, V. R., & Vakayil, A. (2022). SPlit: An Optimal Method for Data
   Splitting. *Technometrics*, 64(2), 166-176.
2. Joseph, V. R. (2022). Optimal Ratio for Data Splitting. *Statistical
   Analysis and Data Mining: The ASA Data Science Journal*, 15(4), 531-538.
3. Vakayil, A., & Joseph, V. R. (2022). Data Twinning. *Statistical
   Analysis and Data Mining: The ASA Data Science Journal*, 15(5), 598-610.
4. Mak, S., & Joseph, V. R. (2018). Support points. *The Annals of
   Statistics*, 46(6A), 2562-2592.
5. Joseph, V. R., & Mak, S. (2021). Supervised compression of big data.
   *Statistical Analysis and Data Mining: The ASA Data Science Journal*,
   14(3), 217-229.
6. Chen, Y., Welling, M., & Smola, A. (2010). Super-Samples from Kernel
   Herding. *UAI*, 109-116.
7. Dwivedi, R., & Mackey, L. (2024). Kernel Thinning. *Journal of Machine
   Learning Research*, 25(152), 1-77.
8. Shetty, A., Dwivedi, R., & Mackey, L. (2022). Distribution compression
   in near-linear time. *ICLR*.
9. Belhadji, A., Sharp, D., & Marzouk, Y. (2025). Weighted quantization
   using MMD: From mean field to mean shift via gradient flows.
   arXiv:2502.10600.
10. Zhang, D., Dai, Q., & Peng, H. (2025). The best instruction-tuning data
    are those that fit. arXiv:2502.04194.
11. Xie, S. M., Santurkar, S., Ma, T., & Liang, P. (2023). Data selection
    for language models via importance resampling (DSIR). *NeurIPS*.
12. Xia, M., Malladi, S., Gururangan, S., Arora, S., & Chen, D. (2024).
    LESS: Selecting influential data for targeted instruction tuning.
    *ICML*, PMLR 235, 54104-54132.
13. Bukharin, A., et al. (2024). Data diversity matters for robust
    instruction tuning (QDIT). *Findings of EMNLP*, 3411-3425.
14. Liu, W., Zeng, W., He, K., Jiang, Y., & He, J. (2024). What makes good
    data for alignment? A comprehensive study of automatic data selection
    in instruction tuning (Deita). *ICLR*.

## Changelog

- 2026-09-03: initial roadmap.
- 2026-09-03: M1 (weighted samples) done; kappa question resolved.
- 2026-09-03: references corrected against publisher metadata (print
  volumes and pages); kernel herding added.
- 2026-09-03: M2 (reference distribution) and `selectrows` done.
- 2026-09-03: M3 (twinning and multiplets) done; high-dimensional nearest-neighbor question resolved.
