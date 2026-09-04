# Kernel thinning backend (roadmap M4)

**Status**: Approved design, pre-implementation
**Date**: 2026-09-04
**Branch**: `feat/kernel-thinning`
**Roadmap**: M4 on the Roadmap page (`docs/src/85-roadmap.md`)
**Builds on**: `docs/superpowers/specs/2026-09-03-twinning-design.md` (M3) and
`2026-09-03-reference-distribution-design.md` (M2)

## TL;DR

Add `KernelThinningSplitter <: AbstractSplitter`: generalized kernel thinning
with the target kernel (Dwivedi & Mackey 2022, Alg. 1, 1a, 1b; kernel halving
from Dwivedi & Mackey 2024, Alg. 2). KT-SPLIT halves the candidate rows `m`
times by probabilistic kernel halving into `2^m` candidate subsets; KT-SWAP
adds a uniform random baseline, keeps the candidate with the smallest MMD to
the target measure, and refines it by one pass of best single-point swaps
against every row of the data. It works with `EnergyKernel` (default) and
`GaussianKernel`, plugs into the `_select_rows`/`_with_kernel` protocol, and
takes `weights`/`reference` through the KT-SWAP objective only. splitiq mirrors
`method='kernel_thinning'` and `delta`. Compress++ is out of scope and moves to
M5. Existing splitters' results are unchanged.

Decisions taken with the user on 2026-09-04:

1. Compress++ is not part of M4. The papers' near-linear time applies to
   root-thinning (`N → 2^g√N`); for split ratios KT is `O(N²)` like herding.
   The roadmap says so and moves Compress++ to M5 (`n ≪ N` selection).
2. `weights` and `reference` enter through the KT-SWAP objective only:
   KT-SPLIT runs on the unweighted candidate rows; KT-SWAP minimizes the MMD
   to `P_w` (or to the reference `P_R`) using herding's weighted/cross data
   terms. Recorded under "Differences from the paper".
3. Default kernel `EnergyKernel()` (the benchmarks' winner); `GaussianKernel`
   is supported. Generalized KT's Theorem 1 covers the energy-distance
   kernel explicitly.
4. Benchmarks: add the two kernel-thinning methods to `benchmark/run.jl`
   (four datasets, N = 10³ and 10⁴) and regenerate the Benchmarks page. No
   scale benchmark.

## The algorithm

Notation: standardized data `X` (`N` rows), kernel `k` (`kernelvalue`), target
measure `P` (the data `P_N`, the weighted data `P_w`, or the reference
`P_R`), output size `n`, all randomness from `rng`.

### Sizes and the candidate rows

`m = ⌊log₂(N / n)⌋` (so `1 ≤ m` whenever `n ≤ N/2`) and `L = n · 2^m ≤ N`.
The rows are shuffled with `rng` and the first `L` of the shuffled order are
the KT-SPLIT input sequence; the remaining `N − L` rows take part only in
KT-SWAP (as swap candidates and through the target measure). When `N / n` is
a power of two and `N` is divisible by `n`, `L = N` and every row enters
KT-SPLIT, which is the papers' setting. `n > N/2` (possible only through
`selectrows`) is an `ArgumentError`: kernel thinning halves, so it selects at
most half of the rows. The shuffle is required by the papers (the input
sequence must be oblivious to the algorithm's randomness) and makes the
excluded rows a uniform random subset.

### Kernel halving (KT 2024, Alg. 2; the split kernel is the target kernel)

Input: sequence `x₁, …, x_ℓ` (indices into `X`), per-step failure probability
`δ_step`, kernel `k`. State: `S₁ = S₂ = ∅`, `σ = 0`. For `i = 1, …, ⌊ℓ/2⌋`
with `(x, x′) = (x_{2i−1}, x_{2i})`:

- `b² = k(x,x) + k(x′,x′) − 2k(x,x′)`.
- `(a, σ) = swap_params(σ, b, δ_step)`:
  `a = max(b σ √(2 log(2/δ_step)), b²)`, then
  `σ² ← σ² + b² · max(0, 1 + (b² − 2a) σ² / a²)`.
- `α = Σ_{j ≤ 2i−2} (k(x_j, x) − k(x_j, x′)) − 2 Σ_{z ∈ S₁} (k(z, x) − k(z, x′))`.
- With probability `min(1, ½ · max(0, 1 − α/a))` swap `x` and `x′`.
- Append `x` to `S₁` and `x′` to `S₂`.

`b = 0` (identical rows) gives `a = 0`; then no swap is made and `σ` is left
unchanged (both assignments are equivalent). A trailing unpaired row (odd
`ℓ`) is dropped from both halves, as in the paper (`⌊ℓ/2⌋` steps). The sum
over the previous rows is threaded over `j` (chunks of `n_threads`) once
`2i − 2` exceeds an internal chunk threshold; cost `O(ℓ²)` kernel evaluations
per halving. Returns `(S₁, S₂)` in append order.

With the energy kernel `k(x, y) = −‖x − y‖`, `b²`, `α`, and every MMD in
this design are differences of kernel values, so they coincide with the
positive-definite form `‖x‖ + ‖y‖ − ‖x − y‖` that the generalized KT paper
uses for the energy distance; `kernelvalue` is used as is.

### KT-SPLIT (GKT Alg. 1a as `m` rounds of kernel halving)

Level `j = 1, …, m` halves each of the `2^{j−1}` sequences of level `j − 1`
with the per-step failure probability `δ_step = δ / (m · L)` (Alg. 1a uses
`δ_i / m` at the step that consumes input point `i`; with the paper's
known-size choice `δ_i = δ / L`, KT 2024 Remark 4, this is constant across
levels). Kernel halving computes `α` as `Σ_{z∈S₂} f(z) − Σ_{z∈S₁} f(z)` with
`f = k(x,·) − k(x′,·)`, which is Alg. 2's `Σ_{j≤2i−2} f(x_j) − 2Σ_{z∈S₁} f(z)`
rewritten. The sums use a fixed chunk size (1,024 rows) whose partial sums
are added in order, so the result does not depend on `n_threads`. The KT paper (Sec. 5.2) states that
KT-SPLIT equals repeated kernel halving, which is how it is implemented
here: it is simpler than the interleaved online form and gives the same
candidates given the same coin flips. Output: `2^m` candidate index vectors
of length `n`, in a fixed order (level-order, `S₁` before `S₂`).

### KT-SWAP (GKT Alg. 1b) with the target measure

Let `d(z) = Σ_l v̄_l k(z, r_l)` be the data term of every row `z` of `X`
against the target measure: the mean over the data rows, the weighted mean
under `weights`, or the mean over the reference rows (weighted by
`reference_weights`). These are exactly herding's four `_data_term` methods,
threaded, `O(N²)` (or `O(NM)` with a reference).

For a candidate `S` of size `n`, up to a constant that does not depend on `S`,

`MMD²(S, P) = (1/n²) Σ_{a,b ∈ S} k(a, b) − (2/n) Σ_{a ∈ S} d(a)`.

1. Baseline: `n` rows drawn uniformly without replacement from `1:N` with
   `rng` (the paper's "standard thinning" of a shuffled sequence).
2. Keep the candidate (baseline included) with the smallest `MMD²(S, P)`.
3. One pass over positions `i = 1, …, n`: with `s = S[i]`, for every row
   `z ∉ S` the change of `MMD²` from replacing `s` by `z` is

   `Δ(z) = (1/n²) [k(z,z) − k(s,s) + 2 (c(z) − k(z,s)) − 2 (c(s) − k(s,s))] − (2/n) (d(z) − d(s))`

   with `c(y) = Σ_{a ∈ S} k(y, a)` maintained for every row `y` of `X`
   (`O(nN)` to build, threaded over `y`). Replace `s` by the `z` with the
   smallest `Δ(z)` if `Δ(z) < 0` (ties to the lowest row index), then
   update `c(y) += k(y, z) − k(y, s)` for all `y` (threaded). Candidates
   are restricted to rows not already in `S`, so the selection stays a
   set of distinct rows; the paper's argmin ranges over all of `S_n`.

`MMD²` never increases during the pass, and the result is never worse than
the baseline (KT 2024, Remark 3). The number of replacements is reported as
`SplitResult.iterations`; `converged` is `true`.

### Cost

KT-SPLIT `O(L²)` kernel evaluations (about `L²/2` over all levels), the
data term `O(N²)`, KT-SWAP `O(nN)`. All three are threaded over the inner
sums. This is the same order as `HerdingSplitter`; the Methods page and the
roadmap say so and name Compress++ (M5) as the near-linear route for
`n ≪ N`.

## Public API

```julia
KernelThinningSplitter(; kernel = EnergyKernel(), ratio = 0.2, delta = 0.5,
                         n_threads = Threads.nthreads(), rng = Random.default_rng())
```

```julia
struct KernelThinningSplitter{K<:SplitKernel,R<:AbstractRNG} <: AbstractSplitter
  kernel::K
  ratio::Float64
  delta::Float64
  n_threads::Int
  rng::R
end
```

- Validation: `0 < ratio < 1`, `0 < delta < 1`, `n_threads > 0`.
- `delta` is the total failure probability `δ` of the KT guarantees; the
  papers' experiments use `δ_i = 1/(2n)`, i.e. `δ = 0.5`.
- `_with_kernel` replaces the kernel (a `:median` bandwidth is resolved by
  `_prepare` as for the other splitters).
- `_select_rows(s::KernelThinningSplitter, kernel, X, n; weights, target,
  target_weights)` returns `(rows, true, swaps)`: the refined coreset in its
  position order, `converged = true`, `iterations = swaps`.
- `show`: `KernelThinningSplitter(kernel=…, ratio=…, delta=…)`.

Internal core (all in `src/kernel_thinning.jl`):

```julia
_swap_params(σ, b, δ) -> (a, σ′)
_kernel_halving(kernel, X, seq::Vector{Int}, δ_step, rng; n_threads) -> (S₁, S₂)
_kt_split(kernel, X, seq, m, δ, rng; n_threads) -> Vector{Vector{Int}}   # 2^m candidates
_kt_swap(kernel, X, candidates, d::Vector{Float64}, rng; n_threads) -> (rows, swaps)
kernel_thinning(kernel, X, n; delta, weights, target, target_weights, n_threads, rng)
  -> (rows, swaps)   # documented, like `herd`
```

`kernel_thinning` resolves the target with `_resolve_target` and the data
term with a new helper `_target_data_term(kernel, X, weights, target,
target_weights, n_threads)` factored out of `herd`'s existing four-way
dispatch (moved, not changed; `herd` calls it too and stays bit-identical).

Python (splitiq): `method='kernel_thinning'` on `datasplit`, `select_rows`,
`multiplet`, with a new keyword `delta: float = 0.5` (only valid with this
method; another method with `delta` set away from the default raises
`ValueError`); `kappa`/`max_iterations`/`tolerance`/`start` with this method
raise `ValueError` as for herding; `n_threads` and `kernel`/`bandwidth` apply.
`SplitResult.method` Literal gains `'kernel_thinning'`.

## Differences from the paper (docstring section and Methods page)

- Target-kernel thinning (the split kernel is the target kernel), not
  square-root KT; justified by Generalized KT, Theorem 1.
- Output size: the papers thin `N` to `⌊N/2^m⌋`; here `n` comes from `ratio`
  (or the caller), `m = ⌊log₂(N/n)⌋`, and only `L = n·2^m` shuffled rows
  enter KT-SPLIT; the rest enter KT-SWAP. Equal to the paper when
  `N/n = 2^m`.
- `weights`/`reference` change only the KT-SWAP objective.
- KT-SWAP candidates exclude rows already in the coreset.
- Compress++ (near-linear time) is not implemented.

## Tests

- `test/test_kernel_thinning.jl` (new):
  - `_swap_params` matches the formula on hand-picked values, including
    `b = 0`.
  - Kernel halving: `S₁ ∪ S₂` is the even-length prefix of `seq`, sizes
    `⌊ℓ/2⌋`; `α` at every step equals the brute-force RKHS inner product
    `Σ_{z∈S₂} f_i(z) − Σ_{z∈S₁} f_i(z)` with `f_i = k(x,·) − k(x′,·)`,
    computed from the halves so far (naive `O(ℓ²)` reference in the test,
    using a recorded coin sequence via a shared `rng`); results are
    reproducible under the same `rng`; threading does not change the
    result (`n_threads = 1` vs `4`).
  - Balance: on a 2-D mixture, the MMD² between `S₁` and the input is below
    the mean over 20 uniform random halvings (both kernels).
  - KT-SPLIT: `2^m` candidates, each of size `n`, partitioning the first
    `n·2^m` rows of the sequence.
  - KT-SWAP: `MMD²` is non-increasing across the pass (recorded through a
    test hook that returns the objective trace, or by recomputing the exact
    MMD before and after) and never above the baseline's; candidates never
    repeat a row; `Δ(z)` for the chosen swap equals the exact difference of
    `mmd(...)` values on a small case.
  - `KernelThinningSplitter`: construction/validation/`show`; `datasplit`
    partition and sizes for `ratio = 0.2` (`m = 2`, `L = 0.8N`) and
    `ratio = 0.25` (`L = N`); `ratio > 0.5` puts the selected rows in train;
    `selectrows` equals the selected side and rejects `n > N/2`;
    reproducibility with `rng`; DataFrame and vector inputs; both kernels;
    `compare` with a mixed list; uniform `weights` reproduce the unweighted
    selection exactly; concentrated `weights` and a sub-population
    `reference` pull the selection (as in the herding tests).
- `test/test_properties.jl` (append): kernel-thinning splits beat random
  splits under the energy distance (`EnergyKernel`) and under MMD
  (`GaussianKernel`) on the 2-D mixture and a 4-D normal — the roadmap's
  acceptance test.
- splitiq `tests/test_kernel_thinning.py` (new): partition and report
  fields (`method`, `kernel`, `bandwidth`), `delta` accepted, error cases,
  `multiplet` with the method, reproducibility with `seed`.

## Benchmarks

`benchmark/run.jl`: add `kernel thinning · energy` and
`kernel thinning · gaussian` to `methods()` (same `rng_seed` pattern),
extend `methods_order`/`colors`/`markers` to six methods, and re-run
(`julia -t auto --project=benchmark benchmark/run.jl`). Regenerate
`results.md`, `quality.png`, `time.png`, `selection.png`. The Benchmarks
page's sections 1–2 are then re-derived from the new table: the "best or
close to best" claim, the per-cell winners table, and the speed sentence
must be rewritten from the numbers (they may change the page's headline);
section 3 (rounding) is untouched.

## Docs

- Methods page: new section "Kernel thinning" (kernel halving, KT-SPLIT,
  KT-SWAP with the target measure, the `(1/n²)…` objective, cost,
  "Differences from the paper"), plus References entries for Dwivedi &
  Mackey (2024, JMLR) and (2022, ICLR).
- Benchmarks page: sections 1–2 re-derived (see above); "How it was run"
  method table gains the two rows.
- Roadmap: M4 done; Current-state row for `KernelThinningSplitter`; the
  M4 text records that split-ratio KT is `O(N²)` and that Compress++ moves
  to M5 (add it to M5's list); changelog line.
- index.md, Python page, splitiq `getting-started.md`/`overview.md`/READMEs:
  the new method and `delta`.
- AGENTS.md gotchas: KT-SPLIT runs on `n·2^m` shuffled rows; `weights`/
  `reference` act on KT-SWAP only; `delta` is the paper's `δ`; swap
  candidates exclude coreset rows; same `O(N²)` class as herding.

## Non-goals

- Compress / Compress++ (M5).
- Square-root or power kernels (target-kernel KT only).
- Weighted KT-SPLIT.
- Multiple KT-SWAP passes.

## References

- Dwivedi, R., & Mackey, L. (2024). Kernel Thinning. *Journal of Machine
  Learning Research*, 25(152), 1–77. Alg. 1, 1a, 1b, 2; Remarks 3–4;
  Sec. 5.2.
- Dwivedi, R., & Mackey, L. (2022). Generalized Kernel Thinning. *ICLR*.
  Alg. 1, 1a, 1b; Theorem 1 (target kernel thinning, energy distance).
- Shetty, A., Dwivedi, R., & Mackey, L. (2022). Distribution Compression in
  Near-linear Time. *ICLR*. (Compress++, deferred to M5.)
