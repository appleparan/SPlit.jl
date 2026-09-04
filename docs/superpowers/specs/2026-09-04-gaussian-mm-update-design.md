# Gaussian-kernel MM update (roadmap M6)

**Status**: Approved design, pre-implementation
**Date**: 2026-09-04
**Branch**: `feat/gaussian-mm-update`
**Roadmap**: M6 on the Roadmap page (`docs/src/85-roadmap.md`)
**Builds on**: `docs/superpowers/specs/2026-09-02-gaussian-kernel-design.md`
(Phase 2a, the Armijo path this replaces) and the stochastic MM of
`2026-09-02-paper-aligned-redesign-design.md` (Phase 1)

## TL;DR

Replace the projected-gradient/Armijo optimizer behind
`support_points(::GaussianKernel, …)` with a majorize-minimize (MM) sweep of
the same shape as the energy-kernel sweep of Mak & Joseph (2018): each point
moves to a kernel-weighted mean of the data (the mean-shift step), pushed by
a linearized repulsion from the other points, divided by a constant that is
always positive. One sweep costs one pass over the data and the point set,
with no line search and no objective evaluation; the objective is
non-increasing at every full-data sweep by the MM argument. Because the
sweep has the energy sweep's structure, the stochastic `kappa` mode of
Joseph & Vakayil (2022) applies unchanged, so `GaussianKernel` gains
`kappa`. Public signatures do not change; `GaussianKernel` results do (a
different optimizer), `EnergyKernel` results stay bit-identical. splitiq
needs no signature change, only a test and a docs line.

Decisions taken with the user on 2026-09-04:

1. The MM sweep replaces the Armijo path outright once the benchmark
   confirms the selected rows are of the same quality; one optimizer per
   kernel, no `update = :mm | :gradient` keyword. The old path survives
   only inside `benchmark/gaussian_update.jl` so the experiment can be
   reproduced.
2. `kappa` is allowed with `GaussianKernel`, with the energy path's
   semantics: an absolute row count, stochastic only when below the number
   of target rows, running-average weight `n0 = 0.2n`. The full-data sweep
   (`alpha = 1`) is the pure MM step.
3. Convergence is the displacement rule only (largest squared per-point
   displacement below `tolerance`), as for `EnergyKernel`; the
   Gaussian-only `rtol` keyword of `support_points` is removed, since the
   sweep never evaluates the objective (evaluating it would double the
   cost of an iteration).
4. Scope: the Julia optimizer and its tests, `benchmark/gaussian_update.jl`
   with its table on the Design experiments page, a rerun of
   `benchmark/run.jl` for the Benchmarks page (the Gaussian support-point
   method changes speed), docs (Methods, Roadmap, README, AGENTS.md), and
   splitiq (a `kappa` + `'gaussian'` test and a docs mention). The version
   bump is a separate PR.

## Why not the paper's update as written

Belhadji, Sharp & Marzouk (2025, arXiv:2502.10600) derive *mean shift
interacting particles* (MSIP) for weighted quantization: the objective is
`F_M(Y) = inf_w MMD²(Σ w_i δ_{y_i}, π)`, the weights `ŵ = K(Y)⁻¹ v₀(Y)` are
re-solved every iteration, and the fixed-point map
`Ψ(Y) = W⁻¹ K̄⁻¹ v̂₁(Y)` needs an `n × n` solve. SPlit's selected subset is
always uniform (AGENTS.md), so the weighted objective is the wrong one,
and an `n × n` solve per iteration is not the cost class of the energy
sweep. The paper's own descent result (Prop. 3.5) only shows that *some*
step-size schedule decreases the MMD, and its experiments use a damping of
`η = 0.5`.

Checked before the design: the damped uniform-weight fixed point (the
paper's eq. 29 with `W = I/n` and the repulsion made explicit) *diverges* on
all four benchmark datasets, because its denominator, data density minus
point-set density at `ξ_m`, crosses zero exactly where the point set fits
the data. The MM update below has the same fixed points but a denominator
that is a sum of two non-negative terms.

## The objective

Data rows `x_1, …, x_N` (standardized), support points `ξ_1, …, ξ_n`,
kernel `k(u, v) = exp(−‖u − v‖² / 2σ²)`. Up to a constant,

```math
f(\xi) = \frac{1}{n^2} \sum_{i,j} k(\xi_i, \xi_j)
  - \frac{2}{nN} \sum_{m,l} \hat w_l\, k(\xi_m, x_l),
```

with `ŵ` the mean-one row weights (`1.0` everywhere when unweighted). This
is `_mmd_objective`, unchanged.

## The majorizer

Write `t = ‖ξ − x‖²`. The data term `−k = −exp(−t/2σ²)` is concave in `t`,
so its tangent at the current point `ξ⁰` is an upper bound:

```math
-k(\xi, x) \le \text{const} + \frac{k(\xi^0, x)}{2\sigma^2}\,\|\xi - x\|^2 .
```

Minimizing this bound alone is the mean-shift step (Fukunaga & Hostetler
1975; the MM view is Fashing & Tomasi 2005).

The repulsion `h(u) = k(u) = exp(−‖u‖²/2σ²)`, `u = ξ_i − ξ_j`, is neither
convex nor concave, but its Hessian `h(u)[uuᵀ/σ⁴ − I/σ²]` has largest
eigenvalue `(h/σ²)(s − 1)` with `s = ‖u‖²/σ²`, maximized at `s = 3`:

```math
L = \max_u \lambda_{\max}\nabla^2 h(u) = \frac{2e^{-3/2}}{\sigma^2} \approx \frac{0.446}{\sigma^2}.
```

So `h(u) ≤ h(u⁰) + ∇h(u⁰)ᵀ(u − u⁰) + (L/2)‖u − u⁰‖²`, and with
`u − u⁰ = δ_i − δ_j` (`δ = ξ − ξ⁰`) and `‖δ_i − δ_j‖² ≤ 2‖δ_i‖² + 2‖δ_j‖²`
the bound separates over points. Summing over the ordered pairs `(m, j)`
and `(j, m)`, `j ≠ m` (the `i = j` terms are the constant `1`), the
majorizer of `f` at `ξ⁰` is `Q(ξ) = Σ_m Q_m(ξ_m) + const` with

```math
Q_m(\xi) = \frac{1}{nN\sigma^2} \sum_l \hat w_l k^0_{ml} \|\xi - x_l\|^2
  + \frac{2}{n^2} \sum_{j \ne m} \Big[ \nabla h(u^0_{mj})^\top (\xi - \xi^0_m)
  + L \|\xi - \xi^0_m\|^2 \Big],
```

where `k⁰_ml = k(ξ⁰_m, x_l)` and `∇h(u⁰_mj) = −k(ξ⁰_m, ξ⁰_j)(ξ⁰_m − ξ⁰_j)/σ²`.
`Q ≥ f` everywhere with equality at `ξ⁰`, and `Q_m` is a quadratic in
`ξ_m` with Hessian `(2/n)(A_m + B) I`, so minimizing it coordinate-wise
under the bounding box is a clamp of the unconstrained minimizer.

## The update

Per point `m`, from the four sums over the data (`s`) and the other points
(`r`):

```text
s0 = Σ_l ŵ_l k(ξ_m, x_l)              s1 = Σ_l ŵ_l k(ξ_m, x_l) x_l
r0 = Σ_{j≠m} k(ξ_m, ξ_j)              r1 = Σ_{j≠m} k(ξ_m, ξ_j) ξ_j

A   = s0 / (N σ²)                      data density at ξ_m
ms  = s1 / s0                          mean-shift target (ms = ξ_m if s0 = 0)
rep = (r0 ξ_m − r1) / (n σ²)           repulsion, = −(1/n) Σ_{j≠m} ∇h(u_mj)
B   = 2 (n − 1) L / n = 4 (n − 1) e^{-3/2} / (n σ²)

ξ_m ← clamp( (A · ms + B · ξ_m + rep) / (A + B), bounds )
```

(`∇Q_m = 0` multiplied by `n/2` gives `A(ξ − ms) + (1/n)Σ∇h + B(ξ − ξ⁰) = 0`.)
`A + B > 0` for `n ≥ 2` (`B > 0`), and for `n = 1` the existing `denom > 0`
guard keeps the point when `A = 0`. Monotone descent of the full-data
sweep: `f(ξ⁺) ≤ Q(ξ⁺) ≤ Q(ξ⁰) = f(ξ⁰)`, clamping included, all points
updated simultaneously (Jacobi order, like `_mm_sweep!`). Fixed points of
the sweep are stationary points of `f`: at a fixed point `∇Q_m = 0` and
`∇Q_m(ξ⁰) = ∇_m f(ξ⁰)` because the majorizer is tangent.

The attraction weight `A/(A + B)` is at most about `0.53` (since
`A ≤ 1/σ²` and `B ≈ 0.89/σ²`), so each sweep moves at most about halfway
to the mean-shift target — the same damping the paper picks empirically.
Per-iteration decrease is therefore smaller than Armijo's, but an
iteration is a single pass. Measured in the design spike at N = 10,000,
n = 2,000, 60 iterations: an MM sweep costs 0.11-0.14 s versus 0.61-0.91 s
for an Armijo iteration (the line search evaluates the objective up to 30
times), and at equal wall time the MM path is level with or ahead of
Armijo on all four benchmark datasets. The benchmark task settles whether
that carries to the selected rows.

### Stochastic mode (`kappa`)

Exactly the energy path's scheme: when `kappa < M` (rows of the target),
each iteration draws `kappa` rows of the target with `_draw_subsample`
(`:uniform`, weights rescaled to mean one within the subsample), computes
`s0`, `s1` and `A = s0/(κσ²)` on the subsample, and blends with the running
constant `Ā_m` (initialized to `0`):

```text
denom = (1 − α) Ā_m + α A + B
ξ_m  ← clamp( ((1 − α) Ā_m ξ_m + α (A · ms + rep) + B ξ_m) / denom, bounds )
Ā_m  ← (1 − α) Ā_m + α A        after the sweep
α    = n0 / (iteration + n0),  n0 = 0.2 n
```

`α = 1` reduces to the full-data update above bit for bit, which is the
invariant the tests pin. `rep` needs no `n_sub/n` rescaling because `A` and
`ms` are already means. The rng is consumed in the same order as on the
energy path (initial points, jitter, one subsample per iteration).

### Weights and reference

`weights` enter through `ŵ` in `s0`/`s1` (mean-one, exactly `1.0` when
unweighted, so the unweighted sums are unchanged); `target`/`target_weights`
replace the data rows by the reference rows `R` and their mean-one weights
via `_resolve_target`, with `N` replaced by `M = size(R, 1)`. Nothing else
in the sweep depends on them.

### Convergence

Largest squared per-point displacement `< tolerance`, checked every
iteration, as for `EnergyKernel`; `converged` and `iterations` report it
honestly. The Gaussian path's second-iteration rule and `rtol` go away with
the line search. A bandwidth far below the row spacing still makes the
objective flat, so the points barely move and the displacement rule stops
at the initial sample — the documented behavior, now reached through the
same rule as the energy kernel.

## Architecture

### `src/optimizer.jl`

- `_mm_sweep!(::EnergyKernel, new_points, current_const, points, sub, sub_w,
  running_const, alpha, bounds, n_threads)` wraps the existing sweep body
  unchanged (its numerics must stay bit-identical); a new
  `_mm_sweep!(k::GaussianKernel{Float64}, …)` with the same signature
  implements the update above: chunked over `n_threads` like the energy
  sweep, one `zeros(p)` pair per task for `s1`/`r1`, explicit coordinate
  loops, no allocation in the inner loops, `current_const[m] = A`.
- `support_points(k::Union{EnergyKernel,GaussianKernel}, data, n; kappa,
  max_iterations, tolerance, n_threads, rng, verbose, weights, target,
  target_weights, _n0_factor, _subsampling)`: the existing energy loop with
  the sweep dispatched on `k`, plus the `isresolved(k)` check. The
  `GaussianKernel` method and its Armijo machinery (`_armijo_step!`,
  `_first_step`, the `rtol` keyword, the second-iteration rule) are
  deleted. `_mmd_objective` and `_mmd_gradient!` stay (`splitquality`/tests
  use the former, the stationarity test the latter).
- `_mmd_trajectory(k, data, n; max_iterations, rng, weights, target,
  target_weights)` becomes the Gaussian twin of `_objective_trajectory`:
  the objective after each full-data sweep.

### `src/splitter.jl`

`SupportPointSplitter` no longer rejects `kappa` with `GaussianKernel`; the
docstring's `kappa`/`tolerance` bullets drop their Gaussian exceptions.
`datasplit`'s kernel resolution is unchanged.

### Removed

`_armijo_step!`, `_first_step`, the `rtol` keyword and the two
"stochastic mode (kappa) is not available for GaussianKernel" errors, with
their tests.

## Testing (properties, not outputs)

`test/test_optimizer.jl`, replacing the Gaussian block:

1. Full-data Gaussian sweep monotonically decreases `_mmd_objective`
   (`_mmd_trajectory`, 40 iterations, N = 150, p = 2, n = 15), also under
   `weights` and under `target`.
2. A fixed point of the sweep is a stationary point: run to a tight
   `tolerance` on a tiny problem, then check that one more sweep moves no
   point by more than `√tolerance` and that `_mmd_gradient!` is small
   relative to the gradient at the initial points.
3. `alpha = 1` sweep equals the full-data sweep bit for bit; `n_threads = 1`
   and `4` give identical points; same `rng` gives identical results.
4. `kappa`: the energy `kappa` block's tests, run on `GaussianKernel(1.0)`:
   stochastic mode runs (`iterations` honest, points within bounds),
   `kappa ≥ N` is the full-data path bit for bit, `kappa = 0` errors, the
   all-zero-weight subsample and `:proportional` checks.
5. Weights as duplication counts equal duplicated rows for one sweep
   (approximate equality, summation order differs).
6. Argument validation: unresolved kernel errors; `rtol` is gone.
7. `test/test_splitter.jl`: the constructor accepts `kappa` with
   `GaussianKernel`; `datasplit` with `GaussianKernel(:median)` and
   `kappa = 200` on N = 2,000 runs, stores the resolved kernel, and beats a
   random split under `mmd` (rng-invariant margin, see the
   `rng-sensitive-tests` note in the developer docs).
8. Existing weighted/reference/multiplet/comparison tests that exercise
   `GaussianKernel` must pass unchanged (they test properties).

splitiq: `test_datasplit.py` gains a `kernel='gaussian', kappa=…` case;
the wrapper passes `kappa` through already.

## Benchmark

`benchmark/gaussian_update.jl`: the four datasets of `benchmark/datasets.jl`
at N = 1,000 and 10,000, n = 0.2N, three seeds, `:median` bandwidth.
Methods: Armijo (the deleted code, carried inside the script), MM
full-data, and MM with `kappa = 1,000` at N = 10,000. Reported per cell:
wall time of `selectrows`, iterations, the exact Gaussian MMD between the
selected rows and the data (N + n ≤ 12,000 rows, below `exact_threshold`),
and the MMD of a uniform random subset. Writes
`docs/src/assets/benchmarks/gaussian_update.md`; a `--quick` mode runs one
seed at N = 1,000. The Design experiments page gets a "Gaussian update
rule" section with the claim, the table link, and the reproduce command.
Then `benchmark/run.jl` is rerun and the Benchmarks page re-derived where
its numbers about `support points · gaussian` change.

## Documentation

- `docs/src/10-methods.md`: the Gaussian section describes the majorizer,
  the update, `kappa`, and the displacement rule; the Armijo paragraphs go.
- `docs/src/85-roadmap.md`: M6 "Done (2026-09-04)" with the outcome; the
  current-state row for `GaussianKernel`; changelog line.
- `docs/src/25-design-experiments.md`: the new section.
- `README.md`, `AGENTS.md`: the Gaussian bullets (no more "no `kappa`
  mode", "first trial step", `rtol`).
- `docs/src/30-python.md` / splitiq docs: `kappa` applies to both kernels.

## References

- Mak, S., & Joseph, V. R. (2018). Support points. *Annals of Statistics*,
  46(6A), 2562-2592.
- Joseph, V. R., & Vakayil, A. (2022). SPlit. *Technometrics*, 64(2),
  166-176.
- Belhadji, A., Sharp, D., & Marzouk, Y. (2025). Weighted quantization using
  MMD: From mean field to mean shift via gradient flows. arXiv:2502.10600.
- Fashing, M., & Tomasi, C. (2005). Mean shift is a bound optimization.
  *IEEE TPAMI*, 27(3), 471-474.
- Fukunaga, K., & Hostetler, L. (1975). The estimation of the gradient of a
  density function, with applications in pattern recognition. *IEEE Trans.
  Inf. Theory*, 21(1), 32-40.

## Amendment (2026-09-04, after the benchmark)

Decision 1's criterion was not met: on `uniform-5d` the MM path's selected
rows are worse than Armijo's (N = 1,000: MMD 0.00267 vs 0.00116; N = 10,000:
0.000393 vs 0.000305, random 0.000401), and on full data the MM sweep never
reaches the displacement rule within the iteration cap while Armijo stops
early by its relative-decrease rule, so its wall time is higher on two of
four datasets at N = 10,000. The `kappa` mode, which only the sweep can
provide, is 3-4x faster than full-data MM at Armijo-level quality on three
datasets (`benchmark/gaussian_update.jl`, Design experiments page). An
over-relaxed sweep (adaptive extrapolation with one objective evaluation
per iteration) was tried and rejected: negligible objective gain at 2-3x the
cost.

Decision taken with the user (option 1 of three): **the Gaussian kernel
keeps the Armijo projected-gradient optimizer on full data and runs the MM
sweep only in stochastic mode (`kappa` below the number of target rows).**
Consequences:

- `support_points(::GaussianKernel, …)` regains `rtol` (full-data rule:
  at least 2 iterations, then displacement or relative decrease) and
  dispatches on `stochastic`; the shared MM loop becomes the internal
  `_support_points_mm`, called by `support_points(::EnergyKernel, …)` and
  by the Gaussian stochastic branch. `kappa ≥ M` is the full-data (Armijo)
  path, bit-identical to `kappa = nothing`.
- `_armijo_step!`, `_first_step`, and the Armijo `_mmd_trajectory` return
  verbatim; the benchmark script's private copy of Armijo goes away and the
  script's `mm` (full-data sweep) arm becomes a private loop over
  `_mm_sweep!`, since that path is no longer reachable through the API.
- `benchmark/rounding.jl` keeps its Armijo loop (its stale
  `_mmd_gradient!` call gains the `w_hat` argument); `benchmark/run.jl` is
  not rerun: the full-data Gaussian path is unchanged, so the Benchmarks
  page stays valid.
- The "one optimizer per kernel" principle is set aside on measured
  grounds, recorded on the Design experiments page.
