# Embedding workflow, Compress++, and the data-selection guide (roadmap M5)

**Status**: Approved design, pre-implementation
**Date**: 2026-09-04
**Branch**: `feat/embedding-workflow`
**Roadmap**: M5 on the Roadmap page (`docs/src/85-roadmap.md`)
**Builds on**: `docs/superpowers/specs/2026-09-04-kernel-thinning-design.md` (M4),
`2026-09-03-twinning-design.md` (M3), `2026-09-03-reference-distribution-design.md` (M2)

## TL;DR

Three additions that turn the splitters into a data-selection toolkit for
embedding matrices:

1. `standardize = false` on `datasplit`, `selectrows`, `multiplet`,
   `splitquality`, and `compare` (Python: `standardize=False`): numeric
   matrices are used as they are, so cosine-normalized embeddings keep their
   geometry. Default `true` keeps every existing result bit-identical.
2. Compress++ (Shetty, Dwivedi & Mackey 2022, Alg. 1–2) inside
   `KernelThinningSplitter(compress = :auto | :always | :never)` and
   `kernel_thinning(...; compress)`: for `n ≪ N` the rows are compressed in
   near-linear time by recursive symmetrized kernel halving to about
   `2^g √N` rows, which kernel thinning then thins to `n`. `:auto` picks it
   when its estimated cost is below plain kernel thinning's and the target
   measure is the data itself.
3. `examples/llm_data_selection.jl`: downloads a public arXiv-abstract
   embedding matrix (10,000 × 384, CC0), cosine-normalizes it, selects
   subsets with the M1–M4 combinations (plain, quality-weighted, targeted at
   a category, Compress++ for `n ≪ N`), and compares the energy distance to
   the (weighted or target) measure against uniform random and K-center
   greedy. Its table is committed under `docs/src/assets/examples/` and
   quoted by a new docs page, "Selecting LLM training data", whose decision
   table (by `N`, `n/N`, weights, target) says which method to use.

Docs: Methods gains "Compress++" and "Skipping preprocessing"; the roadmap
marks M5 done and resolves its two open questions; splitiq mirrors
`standardize` and `compress`.

Decisions taken with the user on 2026-09-04:

1. Compress++ ships as `KernelThinningSplitter(compress = :auto)` (also
   `:always`, `:never`), with `g` chosen internally.
2. Embedding inputs bypass preprocessing through a `standardize = false`
   keyword (a `DataFrame` with `standardize = false` is an `ArgumentError`).
3. The example uses a real public embedding dataset
   (`sondalex/arxiv-abstracts-2021-embeddings-10000` on Hugging Face,
   CC0-1.0: `id`, `content`, `categories`, `embedding`; MiniLM 384-d by
   default, Arctic-large 1024-d selectable).
4. The merged M3/M4 worktrees were removed.

## 1. `standardize`

`_prepare(s, data, weights, reference, reference_weights; standardize)`:

- `standardize = true`: unchanged (fit/apply preprocessing, constant-column
  removal, Helmert encoding).
- `standardize = false`: `data` must be an `AbstractMatrix` or
  `AbstractVector` of reals (a vector becomes an `N × 1` matrix); `X =
  Matrix{Float64}(data)` with no centering, scaling, or column removal. A
  `reference` is converted the same way and must have the same number of
  columns. A `DataFrame` (data or reference) raises
  `ArgumentError("standardize = false needs a numeric matrix or vector; encode DataFrames yourself or keep standardize = true")`.
  `weights`/`reference_weights` are validated as today; a `:median`
  bandwidth is resolved on the raw matrix.

Public keywords (all default `true`): `datasplit(s, data; standardize)`,
`selectrows(s, data, n; standardize)`, `multiplet(s, data, k; standardize)`
(forwarded to every `selectrows`/`_prepare` call), `splitquality(data,
result; standardize)` (scores the raw rows), `compare(methods, data;
standardize)` (forwards to `datasplit` and `splitquality`, and resolves a
`:median` scoring kernel on the raw rows). Python: `standardize: bool = True`
on `datasplit`, `select_rows`, `multiplet`, `splitquality`, `compare`
(passed as the Julia keyword).

This resolves the roadmap's open question on categorical handling in
embedding mode: embedding matrices have no categorical columns, and the
caller keeps the geometry by passing `standardize = false`.

## 2. Compress++

Notation as in the M4 spec; `Xt = permutedims(X)`; all randomness from
`rng`.

### Compress (paper Alg. 1) with kernel-thinning halving

`_compress(kernel, X, seq::Vector{Int}, g::Int, δ_halve, rng; n_threads) -> Vector{Int}`

- If `length(seq) ≤ 4^g`, return `seq`.
- Split `seq` into four consecutive parts of sizes `⌊ℓ/4⌋` or `⌈ℓ/4⌉`
  (the input is already a random permutation, so consecutive parts are
  the paper's "arbitrary subsequences"); recurse on each; concatenate the
  four results; return `HALVE` of the concatenation.
- `HALVE(S)` (paper Ex. 2 and Remark 3, "symmetrized KT"): run
  `kernel_thinning(kernel, X[S, :], ℓ ÷ 2; delta = δ_halve, compress = :never,
  n_threads, rng)` on the block's own rows (`m = 1`, full KT-SWAP against
  the block), then with probability one half (one `rand(rng)`) return the
  selected half, otherwise its complement within `S`. The complement makes
  each halving unbiased (`E[P_HALVE k | S] = P_S k`), which the paper's
  guarantee needs. Rows are returned as indices into `X` in the block's
  order.

The paper assumes `ℓ = 4^k · 4^g`; the uneven four-way split above covers
any `ℓ` (a trailing unpaired row of a HALVE input is dropped by kernel
halving as in the paper, so a block's output is `⌊ℓ/2⌋`). Output size is
about `2^g √N`, up to those roundings.

### Compress++ (paper Alg. 2)

`_compress_plus_plus(kernel, X, n; g, delta, rng, n_threads) -> (rows, swaps)`:

1. `seq = randperm(rng, N)`.
2. `S_C = _compress(kernel, X, seq, g, δ_halve, rng)` with `δ_halve =
   delta / (2 K)`, `K` = the number of HALVE calls Compress makes
   (`K = Σ_{j=0}^{J−1} 4^j`, `J` the recursion depth); the remaining
   `delta / 2` goes to the THIN step. (Union bound; a simplification of the
   paper's per-call schedule in Ex. 6, stated as such in the docstring.)
3. THIN: `(local, swaps) = kernel_thinning(kernel, X[S_C, :], n; delta =
   delta / 2, compress = :never, n_threads, rng)`; return
   `(S_C[local], swaps)`. Kernel thinning's own rules apply inside THIN
   (`m = ⌊log₂(|S_C|/n)⌋`, the complement rule above half), so any
   `n < |S_C|` works. THIN's target measure is `S_C`'s empirical measure,
   as in the paper.

### Choosing `g` and the `:auto` rule

- `g = max(4, ⌈log₂(2n / √N)⌉)`: `4` is the value the paper's experiments
  use throughout; the second term keeps the compressed set at about `2n`
  rows or more. If the realized `|S_C| ≤ n` (rounding at small sizes), `g`
  is increased by one and Compress is rerun (this consumes further `rng`
  draws; it is a guard, not a normal path, and is tested).
- Estimated costs in kernel evaluations: plain kernel thinning
  `≈ 1.5 N²` (halvings, data term, swap pass); Compress++
  `≈ 4^g N (4 log₄ N + 1)` (paper Remark 1 with quadratic HALVE, plus THIN
  on `2^g √N` rows).
- `compress = :auto` uses Compress++ when `weights === nothing &&
  target === nothing` and `4^g (4 log₄ N + 1) < 1.5 N`; otherwise plain
  kernel thinning. `:always` uses it whenever `weights`/`target` are
  `nothing` (an `ArgumentError` otherwise: "Compress++ is defined for the
  data's own distribution; pass compress = :never with weights or a
  reference"); `:never` is today's behavior. `compress` is a field of
  `KernelThinningSplitter` (keyword, default `:auto`) and a keyword of
  `kernel_thinning` (default `:never`, so existing calls are unchanged).

With `:auto` and a split ratio (`n = 0.2N`), `2^g √N ≥ 2n` forces
`4^g ≥ 0.16 N`, and the cost rule then rejects Compress++ for every `N`,
so `datasplit` results with the default splitter stay identical to M4's.
Compress++ only engages through `selectrows`/`multiplet` with `n ≪ N` — at
N = 10⁴ up to n ≈ 800 (`g = 4`), at N = 10⁶ up to a few 10⁴ rows.

`SplitResult.iterations` is THIN's swap count. Documented differences from
the paper: HALVE is kernel thinning of the block (split + swap, `m = 1`);
`δ` is split evenly over the halvings and THIN; uneven four-way splits;
`g` tied to `n`.

## 3. The example

`examples/Project.toml` (SPlit developed from `..`, DuckDB, DataFrames,
Downloads, Statistics, Random, LinearAlgebra, Printf) and
`examples/llm_data_selection.jl`:

1. Download `data/arxiv-abstract-minilm.parquet` (11.6 MB; `--model
   arcticlarge` switches to the 1024-d file) from
   `https://huggingface.co/datasets/sondalex/arxiv-abstracts-2021-embeddings-10000/resolve/main/`
   into `examples/data/` (skipped when present); read `embedding`,
   `categories`, `content` with DuckDB (`read_parquet`).
2. `E`: rows cosine-normalized (`x / ‖x‖`). `N = 10,000`, `p = 384`.
3. Quality weights `w`: abstract length in characters, clipped at the 99th
   percentile (a stand-in for a quality score). Target `R`: the rows whose
   `categories` contain `cs.LG` (machine learning), a few hundred rows.
4. Selections of `n = 1,000` rows with `standardize = false`, each timed:
   - `random` (uniform, mean of 5 seeds),
   - `k-center greedy` (farthest-first traversal from a random row; Sener &
     Savarese 2018 use it as the core-set baseline; implemented in the
     script, `O(nNp)`),
   - `HerdingSplitter(EnergyKernel())`, `TwinningSplitter()`,
     `KernelThinningSplitter()`, `SupportPointSplitter(kappa = 1_000)`,
   - the same four with `weights = w` (twinning skipped: it rejects
     weights),
   - the same four with `reference = R` (twinning skipped),
   - `KernelThinningSplitter(compress = :always)` and `(:never)` for
     `n = 300` (`n ≪ N`), timed against each other.
5. Scores with `standardize = false`: energy distance of the selection to
   the plain data (`energydistance(E[sel, :], E)`), to the weighted data
   (`weights_y = w`), and to the target (`energydistance(E[sel, :], R)`),
   whichever the setting optimizes, plus the plain one for every setting.
6. Prints a markdown table and writes it to
   `docs/src/assets/examples/llm_selection.md` (committed, like the
   benchmark tables; `--out` overrides). Runtime a few minutes on the
   benchmark machine; not run in CI.

## 4. Docs

- New page `docs/src/40-llm-data-selection.md`, "Selecting LLM training
  data": the workflow (embed → cosine-normalize → `standardize = false` →
  pick the measure: plain, `weights`, `reference` → pick the method →
  check with `splitquality`/`energydistance`), a decision table

  | N | n / N | weights or target? | method |
  |---|---|---|---|
  | ≤ 10⁴ | any | any | `HerdingSplitter(EnergyKernel())` (exact, fastest); `KernelThinningSplitter` when MMD is the criterion |
  | 10⁵–10⁶ | split ratio (≥ 0.1) | no | `TwinningSplitter` (seconds to minutes) |
  | 10⁵–10⁶ | split ratio | yes | `HerdingSplitter` or `KernelThinningSplitter` (`O(N²)`, minutes to an hour) |
  | ≥ 10⁵ | `n ≪ N` (≤ a few % ) | no | `KernelThinningSplitter(compress = :auto)` (near-linear) |
  | ≥ 10⁵ | `n ≪ N` | yes | `HerdingSplitter` with `weights`/`reference` |

  filled with the example's numbers and the Benchmarks pages' timings, a
  Julia and a Python snippet, and a "what this does not settle" paragraph
  on weighted energy distance as a combination rule (the example is the
  first measurement; stratified selection by quality quantile remains the
  alternative).
- Methods page: "Compress++" (Compress recursion, symmetrized halving, `g`,
  the `:auto` rule, differences from the paper) and "Skipping
  preprocessing" (`standardize = false`, what changes in step 1, when to
  use it).
- Roadmap: M5 done; Current-state rows for `standardize` and Compress++;
  both open questions resolved with one-line notes; changelog.
- index.md quick start gains a `standardize = false` line; Python page and
  splitiq docs gain `standardize` and `compress`; READMEs; AGENTS.md gotchas
  (`standardize = false` skips constant-column removal too; Compress++
  only for the data's own measure; `:auto` never triggers at split
  ratios; `g = max(4, ⌈log₂(2n/√N)⌉)`).

## 5. Tests

- `test/test_preprocessing.jl` or a new `test/test_standardize.jl`:
  `standardize = false` gives `X == Matrix{Float64}(data)` through
  `datasplit`/`selectrows` (checked via a selection that is identical to
  calling the splitter core on the raw matrix); a vector becomes `N × 1`;
  DataFrame errors; `splitquality(...; standardize = false)` equals
  `mmd`/`energydistance` on the raw rows; `compare(...; standardize =
  false)` runs; `multiplet` forwards it; a constant column survives (no
  removal); default `true` unchanged.
- `test/test_kernel_thinning.jl` (append): `_compress` returns distinct
  rows of `seq`, size within the rounding band of `2^g √N` (assert
  `2^g √N / 2 ≤ |S_C| ≤ 2 · 2^g √N` on a few `N`), base case returns the
  input, deterministic under `rng`; symmetrization: across seeds both the
  half and its complement occur; `kernel_thinning(...; compress =
  :always)` returns `n` distinct rows and beats uniform random subsets on
  the 2-D mixture at `n ≪ N` (energy distance); `:auto` equals `:never`
  when the cost rule says plain (split ratio) and equals `:always` when it
  says compress (`n ≪ N`, checked through the predicate function
  `_compress_pays_off(N, n)` and by output equality under the same rng);
  `:always` with `weights`/`target` errors; `:auto` with `weights` equals
  `:never`; the `g` guard path (a tiny `N` where `|S_C| ≤ n` forces a
  retry) terminates with `n` rows.
- `test/test_properties.jl`: Compress++ selections beat random under the
  energy distance on the mixture at `n = 200` from `N = 8,000`.
- splitiq `tests/test_standardize.py` and additions to
  `test_kernel_thinning.py`: `standardize=False` on all five functions
  (DataFrame → `ValueError`), `compress='auto'|'always'|'never'` and the
  error case.

## Non-goals

- Running the example in CI or committing the downloaded data.
- Compress++ with `weights`/`reference` (paper defines it for the input's
  own distribution).
- Exposing `g`.
- A K-center splitter type (baseline lives in the example only).

## References

- Shetty, A., Dwivedi, R., & Mackey, L. (2022). Distribution Compression in
  Near-linear Time. *ICLR*. Alg. 1 (Compress), Alg. 2 (Compress++),
  Remark 1 (runtime), Remark 3 (symmetrization), Ex. 6 (KT-Compress++),
  Sec. 5 (`g = 4`).
- Dwivedi, R., & Mackey, L. (2022). Generalized Kernel Thinning. *ICLR*.
- Sener, O., & Savarese, S. (2018). Active learning for convolutional
  neural networks: a core-set approach. *ICLR*. (K-center greedy baseline.)
- Dataset: `sondalex/arxiv-abstracts-2021-embeddings-10000` (Hugging Face,
  CC0-1.0), a subset of `gfissore/arxiv-abstracts-2021` embedded with
  `sentence-transformers/all-MiniLM-L6-v2` and Snowflake Arctic models.
