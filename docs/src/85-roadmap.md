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

SPlit.jl is now a distribution-preserving subset selection library, and
one interface serves both audiences the roadmap set out to reach:

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
| `SupportPointSplitter` with `GaussianKernel` | done | Minimizes squared MMD by projected gradient descent with Armijo backtracking on full data; `kappa` runs a mean-shift MM sweep on subsamples (roadmap M6). A `:median` bandwidth is resolved at `datasplit` time and the resolved kernel is stored in `result.method.kernel`. |
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
| `KernelThinningSplitter` | done | Target-kernel KT (Dwivedi & Mackey, 2022, 2024): kernel halving, KT-SPLIT, KT-SWAP; energy or Gaussian kernel; `O(N²)` like herding. |
| k-fold splitting (`multiplet`) | done | Strategies S1/S2/S3 of the twinning paper; S1/S2 work with every splitter. |
| Skipping preprocessing (`standardize = false`) | done | `datasplit`, `selectrows`, `multiplet`, `splitquality`, `compare`: the numeric matrix is used unchanged, with no encoding, constant-column removal, or standardization; `DataFrame` input is rejected. See [Methods](@ref methods). |
| Compress++ | done | `KernelThinningSplitter(compress = :auto)` (Shetty, Dwivedi & Mackey, 2022): near-linear kernel thinning for `n ≪ N` against the data's own measure; see [Methods](@ref compress). |
| LLM data-selection workflow | done | Embedding matrices end to end: the example under `examples/` and the [decision table](@ref llm-data-selection). |
| Time-series windows | done (example and docs) | Flatten fixed-length windows into rows (variable-major), standardize per variable, and select with `standardize = false`; see [Time-series windows](@ref time-series). Measured on a synthetic two-regime series: window-level selection separates regimes that point-level statistics cannot, twinning/herding/kernel thinning beat random once `L` reaches the dependence length, and support points do not at `L*p` in the hundreds. Grouped selection (windows from the same event) and rolling-origin selection are not implemented. |
| High-dimensional data (p in the hundreds) | partly measured | Twinning measured at p = 768 (N = 10³-10⁵) on the [Design experiments](@ref twinning-trees) page; the search structure switches by dimension, and above `TWINNING_BRUTE_FORCE_DIMENSION`/`NEAREST_BRUTE_FORCE_DIMENSION` runs the plain-matrix search ([Design experiments](@ref matrix-brute-force)), which compiles once for any width. Support-point and herding splitters remain untested above p = 10. The [time-series contrast](@ref time-series) adds a second data point: twinning's ratio to random rises from 0.38 at 24 columns to 0.89 at 3,072, support points reach parity with random by 1,536, and twinning and support points now run at 12,288 columns — the compile ceiling that used to fail there is gone. |

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

Done (2026-09-04): added `KernelThinningSplitter <: AbstractSplitter`
implementing generalized kernel thinning with the target kernel (Dwivedi
& Mackey, 2022, 2024): kernel halving, KT-SPLIT, and KT-SWAP against the
target measure, under `EnergyKernel` or `GaussianKernel`. Compress++ is
not included: it applies to `n ≈ √N` root-thinning and moves to M5.

- Output size is a power of two; document how arbitrary `n` is handled.
- Tests: on a Gaussian mixture, MMD is significantly below a uniform
  random subset.

Why: selects directly from the data, with no continuous optimization and
no nearest-neighbor assignment step, and comes with an MMD rate of
O(sqrt(log n / n)) that neither the support-point splitters nor Twinning
provide. Near-linear time (Compress++) applies to `n ≈ √N` root-thinning and
was delivered in M5 as `compress = :auto`; at the default split ratio the
cost is `O(N²)`, the herding class.

### M5: embedding workflow, docs, and comparison

Done (2026-09-04). Four deliverables:

- `standardize = false` on `datasplit`, `selectrows`, `multiplet`,
  `splitquality`, and `compare`, so an embedding matrix goes through
  unchanged; see [Methods](@ref methods).
- Compress++ (Shetty, Dwivedi & Mackey, 2022) as
  `KernelThinningSplitter(compress = :auto | :always | :never)`; see
  [Methods](@ref compress).
- `examples/llm_data_selection.jl`, comparing every splitter against
  uniform random and K-center greedy on 5,000 arXiv-abstract embeddings
  under the plain, weighted, and targeted measures. It is not run in CI;
  its table is committed under `docs/src/assets/examples/`.
- The [Selecting LLM training data](@ref llm-data-selection) page: the
  workflow and the decision table.

Why last: it needs all of M1-M4 at once. It is also where the reframing
of the package landed — the top-level docs now describe subset selection,
with train/test splitting as one of its entry points.

### M6: MMD gradient-flow update

Done (2026-09-04). Added a majorization-minimization sweep to the
Gaussian-kernel path (mean-shift data term, majorized repulsion,
structurally the energy sweep of Mak & Joseph, 2018) and `kappa` for
`SupportPointSplitter` with `GaussianKernel`, with the stochastic
semantics, running-average blend, and displacement rule of `EnergyKernel`.
Benchmarked against the existing Armijo path (`benchmark/gaussian_update.jl`),
the sweep was kept only for stochastic mode: on full data it never reaches
the displacement rule within the iteration cap and its selected rows are
worse on `uniform-5d` at both sizes and on `t3-3d` at N = 1,000 (0.000156
vs 0.000118), level or better elsewhere. In `kappa` mode, where only the
sweep is affordable,
the selected-row MMD at N = 10,000 is within about 3% of Armijo's on
`normal-10d` and `t3-3d`, about 29% higher on `uniform-5d` (0.000393 vs
0.000305), and about 10x higher on `mixture-2d` (5.47e-6 vs 5.6e-7, still
about 2x below the random subset's 1.23e-5), at 3.4-3.8x lower wall time
than the full-data sweep (see [Design experiments](@ref gaussian-update)).
The weighted mean-shift map of Belhadji, Sharp & Marzouk (2025) was not
adopted as written because it re-solves the subset's weights every
iteration and the package's selected subset is uniform.

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
  apply. Resolved 2026-09-04 by `standardize = false`, which bypasses
  preprocessing entirely: embedding matrices carry no categorical columns,
  so no separate entry point is needed, and a `DataFrame` is rejected
  rather than silently encoded.
- Is weighted energy distance the right combination rule? Combining a
  quality score with distribution matching via a weighted empirical
  distribution is natural but not validated in the literature. First
  measured 2026-09-04 in `examples/llm_data_selection.jl` (see the
  [LLM data-selection page](@ref llm-data-selection)): the weighted
  selections track the weighted corpus better than the unweighted ones do.
  Left open pending downstream results; if it underperforms, alternatives
  include stratified selection by quality quantile.
- Nearest-neighbor structures above a few thousand columns. Resolved
  2026-09-05 by the plain-matrix `MatrixSearch` on the
  [Design experiments](@ref matrix-brute-force) page: it compiles once for
  any width and replaces `BruteTree` in twinning (`TWINNING_BRUTE_FORCE_DIMENSION`
  stays 50) and the k-d tree in `select_nearest` above the new
  `NEAREST_BRUTE_FORCE_DIMENSION = 200`; it also removes the compile-time
  ceiling that failed at `L*p = 12,288` in the
  [time-series example](@ref time-series).
- Grouped and rolling-origin selection for time series. Not implemented:
  there is no constraint to keep windows from the same event or recording
  together, and no dedicated support for rolling-origin windows, where
  representing each origin as a flattened window is the wrong
  representation. The [time-series page](@ref time-series) documents the
  origin-level state-vector workaround as a substitute.

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
15. Combes, F., Fraiman, R., & Ghattas, B. (2022). Time Series Sampling.
    *Engineering Proceedings*, 18(1), 32.
16. Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D.,
    & Jones, N. S. (2019). catch22: CAnonical Time-series CHaracteristics.
    *Data Mining and Knowledge Discovery*, 33(6), 1821-1852.
17. Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., & Xu, B.
    (2022). TS2Vec: Towards Universal Representation of Time Series.
    *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(8),
    8980-8987.

## Changelog

- 2026-09-03: initial roadmap.
- 2026-09-03: M1 (weighted samples) done; kappa question resolved.
- 2026-09-03: references corrected against publisher metadata (print
  volumes and pages); kernel herding added.
- 2026-09-03: M2 (reference distribution) and `selectrows` done.
- 2026-09-03: M3 (twinning and multiplets) done; high-dimensional nearest-neighbor question resolved.
- 2026-09-04: M4 (kernel thinning) done; Compress++ moved to M5.
- 2026-09-04: M5 (embedding workflow, Compress++, data-selection guide)
  done; both remaining open questions resolved.
- 2026-09-04: M6 (Gaussian MM sweep in `kappa` mode, `kappa` for
  `GaussianKernel`) done.
- 2026-09-05: time-series window flattening example and docs page done;
  high-dimensional data row updated with the `L*p` dimension-ladder
  measurement; two open questions added (nearest-neighbor structures above
  a few thousand columns, grouped and rolling-origin selection);
  references 15-17 added.
- 2026-09-05: `MatrixSearch` (issue #72) replaces the static-vector
  brute-force structure above `TWINNING_BRUTE_FORCE_DIMENSION`/
  `NEAREST_BRUTE_FORCE_DIMENSION`; nearest-neighbor structures above a few
  thousand columns question resolved; high-dimensional data row updated.
