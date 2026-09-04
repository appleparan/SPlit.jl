# Gaussian MM Update (M6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Armijo projected-gradient optimizer behind `support_points(::GaussianKernel, …)` with a majorize-minimize sweep of the energy sweep's shape, which gives `GaussianKernel` the stochastic `kappa` mode; benchmark it against the old path; document it; mirror the `kappa` change in splitiq.

**Architecture:** `_mm_sweep!` gains a kernel-dispatched first argument: the energy method wraps the existing sweep body unchanged, a new `GaussianKernel{Float64}` method implements the mean-shift-plus-majorized-repulsion update. One `support_points` loop serves both kernels (the current energy loop with `_mm_sweep!(k, …)`), so `kappa`, `weights`, `target`, and the displacement convergence rule are shared. The Armijo helpers, the Gaussian `rtol` keyword, and the two "kappa not available" errors are deleted; the old loop survives only inside `benchmark/gaussian_update.jl`.

**Tech Stack:** Julia 1.10+ (CI tests on 1.12), Documenter.jl, CairoMakie (benchmark), Python 3.13 + juliacall (splitiq), pytest, uv, pre-commit (JuliaFormatter 1.0.62, markdownlint, ruff).

**Spec:** `docs/superpowers/specs/2026-09-04-gaussian-mm-update-design.md`

## Global Constraints

- `EnergyKernel` results stay bit-identical: the body of the existing energy sweep is not edited, only wrapped.
- Public signatures unchanged. `SupportPointSplitter` accepts `kappa` with any kernel. `support_points(::GaussianKernel)` loses its `rtol` keyword and gains `kappa`, `_n0_factor`, `_subsampling` with the energy semantics.
- The Gaussian update per point `m` (spec, "The update"): `A = s0/(n_sub σ²)`, `ms = s1/s0` (`ms = ξ_m` when `s0 = 0`), `rep = (r0 ξ_m − r1)/(n σ²)`, `B = 4(n−1)e^{−3/2}/(n σ²)`, `ξ_m ← clamp((A·ms + B·ξ_m + rep)/(A + B))`; stochastic blend `denom = (1−α)Ā_m + αA + B`, numerator `(1−α)Ā_m ξ_m + α(A·ms + rep) + B ξ_m`, `Ā ← (1−α)Ā + αA` after the sweep, `α = n0/(iteration + n0)`, `n0 = 0.2n`.
- Convergence: largest squared per-point displacement `< tolerance` only; `converged`/`iterations` honest.
- `_mm_sweep!` inner loops allocate nothing; explicit coordinate loops; results independent of `n_threads`.
- All randomness through the caller's `rng`; nothing in `src/` seeds or prints on a default path. Never cite or compare with other implementations. Docstrings sit directly above what they document.
- Tests encode properties (monotone descent, stationarity, bit-identity, beats random); no output matching. Existing tests are edited only where they pin the removed behavior (the two `kappa` errors, `rtol`, the "accepted steps" wording).
- Every Julia capability lands in splitiq in this branch (test + docs mention).
- Commands: one test file `julia --project=<worktree> <worktree>/test/<file>.jl`; the suite `julia --project=<worktree> -e "using Pkg; Pkg.test()"`; changed test files also on `julia +1.12 --project=<worktree> <file>`. Commit messages `<type>: <Capitalized description>` + trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; pre-commit runs on commit, never bypass it.
- Work only in `/home/appleparan/src/SPlit.jl/.claude/worktrees/feat-gaussian-mm-update` (branch `feat/gaussian-mm-update`). Use absolute paths. Do not run `benchmark/run.jl` or `benchmark/rounding.jl` unless the task says so.

---

## File structure

| File | Responsibility |
|---|---|
| `src/optimizer.jl` | `_mm_sweep!(::EnergyKernel, …)` wrapper, `_mm_sweep!(::GaussianKernel{Float64}, …)`, one `support_points(::Union{EnergyKernel,GaussianKernel}, …)`, `_mmd_trajectory` on the sweep; Armijo helpers deleted |
| `src/splitter.jl` | constructor accepts `kappa` with `GaussianKernel`; docstring |
| `test/test_optimizer.jl`, `test/test_splitter.jl`, `test/test_properties.jl` | tests |
| `splitiq/tests/test_datasplit.py`, `docs/src/30-python.md`, `splitiq/docs/*.md` | parity |
| `benchmark/gaussian_update.jl` (new), `docs/src/assets/benchmarks/gaussian_update.md` (new) | Armijo vs MM experiment |
| `benchmark/rounding.jl` | local Gaussian loop switched to the sweep |
| `docs/src/assets/benchmarks/results.md`, `quality.png`, `time.png`, `selection.png`, `rounding.md`, `rounding.png` | regenerated |
| `docs/src/10-methods.md`, `25-design-experiments.md`, `20-benchmarks.md`, `85-roadmap.md`, `README.md`, `AGENTS.md` | docs |

---

### Task 1: Gaussian MM sweep

**Files:**

- Modify: `src/optimizer.jl` (add after the existing `_mm_sweep!`, which ends at the line `return nothing` / `end` before `# Validate the (weights | target, target_weights) combination`)
- Test: `test/test_optimizer.jl` (append a new top-level testset at the end of the file)

**Interfaces:**

- Consumes: the existing `_mm_sweep!(new_points, current_const, points, subsample_data, subsample_weights, running_const, alpha, bounds, n_threads)` (energy), `_mmd_objective(k, points, data[, w_bar])`, `_mmd_gradient!(G, k, points, data, w_hat, n_threads)`, `_data_bounds`, `_initial_points`, `_mean_one_weights`.
- Produces:
  - `_mm_sweep!(::EnergyKernel, new_points, current_const, points, subsample_data, subsample_weights, running_const, alpha, bounds, n_threads)` — forwards to the existing sweep.
  - `_mm_sweep!(k::GaussianKernel{Float64}, new_points::Matrix{Float64}, current_const::Vector{Float64}, points::Matrix{Float64}, subsample_data::AbstractMatrix{Float64}, subsample_weights::AbstractVector{Float64}, running_const::Vector{Float64}, alpha::Float64, bounds::Matrix{Float64}, n_threads::Int) -> nothing` — writes `new_points` and `current_const[m] = A_m`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_optimizer.jl`:

```julia
@testset "Gaussian MM sweep" begin
  k = GaussianKernel(1.0)

  # One full-data sweep from the current points, alpha = 1.
  function sweep(k, points, data, w_hat; n_threads = 1)
    n = size(points, 1)
    new_points = similar(points)
    current_const = zeros(n)
    running_const = zeros(n)
    SPlit._mm_sweep!(
      k,
      new_points,
      current_const,
      points,
      data,
      w_hat,
      running_const,
      1.0,
      SPlit._data_bounds(data),
      n_threads,
    )
    return new_points, current_const
  end

  @testset "one sweep never increases the objective" begin
    for seed = 1:5
      rng = MersenneTwister(500 + seed)
      data = randn(rng, 120, 3)
      points = data[rand(rng, 1:120, 12), :] .+ 0.05 .* randn(rng, 12, 3)
      new_points, _ = sweep(k, points, data, ones(120))
      @test SPlit._mmd_objective(k, new_points, data) <=
            SPlit._mmd_objective(k, points, data) + 1e-12
      w = rand(rng, 120) .^ 2
      w_hat = SPlit._mean_one_weights(w)
      w_bar = w ./ sum(w)
      new_w, _ = sweep(k, points, data, w_hat)
      @test SPlit._mmd_objective(k, new_w, data, w_bar) <=
            SPlit._mmd_objective(k, points, data, w_bar) + 1e-12
    end
  end

  @testset "current_const holds the data density A = Σ ŵ k / (N σ²)" begin
    rng = MersenneTwister(510)
    data = randn(rng, 50, 2)
    points = randn(rng, 4, 2)
    _, c = sweep(k, points, data, ones(50))
    for m = 1:4
      expected =
        sum(SPlit.kernelvalue(k, view(points, m, :), view(data, l, :)) for l = 1:50) / 50
      @test isapprox(c[m], expected; rtol = 1e-12)
    end
  end

  @testset "threaded sweep equals serial sweep bit for bit" begin
    rng = MersenneTwister(520)
    data = randn(rng, 200, 4)
    points = randn(rng, 17, 4)
    a, ca = sweep(k, points, data, ones(200); n_threads = 1)
    b, cb = sweep(k, points, data, ones(200); n_threads = 4)
    @test a == b && ca == cb
  end

  @testset "weights as duplication counts equal duplicated rows" begin
    rng = MersenneTwister(530)
    base = randn(rng, 40, 2)
    counts = rand(rng, 1:3, 40)
    dup = vcat([repeat(base[i:i, :], counts[i]) for i = 1:40]...)
    points = randn(rng, 6, 2)
    a, _ = sweep(k, points, base, SPlit._mean_one_weights(Float64.(counts)))
    b, _ = sweep(k, points, dup, ones(size(dup, 1)))
    @test isapprox(a, b; rtol = 1e-10)
  end

  @testset "points stay inside the bounding box" begin
    rng = MersenneTwister(540)
    data = rand(rng, 80, 2)
    points = rand(rng, 5, 2) .* 4 .- 2      # start outside [0, 1]²
    new_points, _ = sweep(k, points, data, ones(80))
    @test all(0 .<= new_points .<= 1)
  end

  @testset "a fixed point of the sweep is a stationary point of the objective" begin
    rng = MersenneTwister(550)
    data = randn(rng, 60, 2)
    points = data[1:4, :] .+ 0.1 .* randn(rng, 4, 2)
    G0 = similar(points)
    SPlit._mmd_gradient!(G0, k, points, data, ones(60), 1)
    for _ = 1:500
      points, _ = sweep(k, points, data, ones(60))
    end
    again, _ = sweep(k, points, data, ones(60))
    @test maximum(abs, again .- points) < 1e-9
    G = similar(points)
    SPlit._mmd_gradient!(G, k, points, data, ones(60), 1)
    @test maximum(abs, G) < 1e-6 * maximum(abs, G0)
  end

  @testset "energy wrapper forwards to the energy sweep" begin
    rng = MersenneTwister(560)
    data = randn(rng, 50, 2)
    points = randn(rng, 5, 2)
    n = 5
    a = similar(points)
    b = similar(points)
    ca = zeros(n)
    cb = zeros(n)
    bounds = SPlit._data_bounds(data)
    SPlit._mm_sweep!(a, ca, points, data, ones(50), zeros(n), 1.0, bounds, 1)
    SPlit._mm_sweep!(EnergyKernel(), b, cb, points, data, ones(50), zeros(n), 1.0, bounds, 1)
    @test a == b && ca == cb
  end
end
```

- [ ] **Step 2: Run the file to verify it fails**

Run: `julia --project=/home/appleparan/src/SPlit.jl/.claude/worktrees/feat-gaussian-mm-update /home/appleparan/src/SPlit.jl/.claude/worktrees/feat-gaussian-mm-update/test/test_optimizer.jl`
Expected: the new testset errors with `MethodError: no method matching _mm_sweep!(::GaussianKernel{Float64}, …)`; every earlier testset still passes.

- [ ] **Step 3: Implement the two methods**

Insert directly after the existing energy `_mm_sweep!` (before the `# Validate the (weights | target, target_weights) combination` comment):

```julia
# Kernel-dispatched entry point; the energy body above is unchanged so its
# results stay bit-identical.
_mm_sweep!(::EnergyKernel, args...) = _mm_sweep!(args...)

# Gaussian-kernel MM sweep (design record: 2026-09-04-gaussian-mm-update).
# The data term −k(ξ, x) is concave in ‖ξ − x‖², so its tangent majorizer
# gives the mean-shift step; the repulsion k(ξ_m, ξ_o) is majorized by its
# L-smooth quadratic bound with L = 2e^{-3/2}/σ² (the largest Hessian
# eigenvalue of a Gaussian), split over the two points. Per point m:
#   A   = Σ_i ŵ_i k(ξ_m, x_i) / (n_sub σ²)             data density
#   ms  = Σ_i ŵ_i k(ξ_m, x_i) x_i / Σ_i ŵ_i k(ξ_m, x_i)  mean-shift target
#   rep = Σ_{o≠m} k(ξ_m, ξ_o) (ξ_m − ξ_o) / (n σ²)      linearized repulsion
#   B   = 2 (n − 1) L / n = 4 (n − 1) e^{-3/2} / (n σ²)
#   ξ_m ← clamp((A ms + B ξ_m + rep) / (A + B), bounds)
# The full-data sweep (alpha = 1) never increases the objective. In
# stochastic mode `alpha` blends A and the data numerator with the running
# constant exactly as the energy sweep does, so the loop in
# `support_points` is shared. `current_const[m]` receives A.
function _mm_sweep!(
  k::GaussianKernel{Float64},
  new_points::Matrix{Float64},
  current_const::Vector{Float64},
  points::Matrix{Float64},
  subsample_data::AbstractMatrix{Float64},
  subsample_weights::AbstractVector{Float64},
  running_const::Vector{Float64},
  alpha::Float64,
  bounds::Matrix{Float64},
  n_threads::Int,
)
  n, p = size(points)
  n_sub = size(subsample_data, 1)
  s2 = k.bandwidth^2
  inv2s2 = 1 / (2 * s2)
  B = 4 * (n - 1) * exp(-1.5) / (n * s2)
  chunks = collect(Iterators.partition(1:n, cld(n, max(1, n_threads))))
  @sync for chunk in chunks
    Threads.@spawn begin
      s1 = zeros(p)
      r1 = zeros(p)
      for m in chunk
        s0 = 0.0
        fill!(s1, 0.0)
        for i = 1:n_sub
          d = 0.0
          for j = 1:p
            d += (subsample_data[i, j] - points[m, j])^2
          end
          w = subsample_weights[i] * exp(-d * inv2s2)
          s0 += w
          for j = 1:p
            s1[j] += w * subsample_data[i, j]
          end
        end
        r0 = 0.0
        fill!(r1, 0.0)
        for o = 1:n
          o == m && continue
          d = 0.0
          for j = 1:p
            d += (points[m, j] - points[o, j])^2
          end
          w = exp(-d * inv2s2)
          r0 += w
          for j = 1:p
            r1[j] += w * points[o, j]
          end
        end
        A = s0 / (n_sub * s2)
        current_const[m] = A
        denom = (1 - alpha) * running_const[m] + alpha * A + B
        for j = 1:p
          ms = s0 > 0 ? s1[j] / s0 : points[m, j]
          rep = (r0 * points[m, j] - r1[j]) / (n * s2)
          x = if denom > 0
            (
              (1 - alpha) * running_const[m] * points[m, j] +
              alpha * (A * ms + rep) +
              B * points[m, j]
            ) / denom
          else
            points[m, j]
          end
          new_points[m, j] = clamp(x, bounds[j, 1], bounds[j, 2])
        end
      end
    end
  end
  return nothing
end
```

- [ ] **Step 4: Run the file to verify it passes**

Run: `julia --project=/home/appleparan/src/SPlit.jl/.claude/worktrees/feat-gaussian-mm-update /home/appleparan/src/SPlit.jl/.claude/worktrees/feat-gaussian-mm-update/test/test_optimizer.jl`
Expected: all pass. If the stationarity ratio `maximum(abs, G) / maximum(abs, G0)` is not below `1e-6` after 500 sweeps, report the ratio and the displacement of the extra sweep in the task report instead of loosening the threshold; do the same for the `1e-9` displacement bound.

- [ ] **Step 5: Commit**

```bash
git add src/optimizer.jl test/test_optimizer.jl
git commit -m "feat: Add the Gaussian-kernel MM sweep

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: One `support_points` loop for both kernels, Armijo removed, `kappa` for Gaussian

**Files:**

- Modify: `src/optimizer.jl` — the energy `support_points` docstring and method (`"""\n    support_points(kernel, data, n; kwargs...)` … `end` of the method), delete `_mmd_gradient!`'s neighbours `_armijo_step!`, `_first_step`, the whole Gaussian `support_points` docstring + method, and rewrite `_mmd_trajectory`
- Modify: `src/splitter.jl:31-42` (docstring bullets) and `:78-79` (the constructor error)
- Test: `test/test_optimizer.jl` (edit the `support_points with GaussianKernel` testset; the rest is append-only), `test/test_splitter.jl:85-88`, `test/test_properties.jl` (append)

**Interfaces:**

- Consumes: Task 1's `_mm_sweep!(k, …)`.
- Produces:
  - `support_points(k::Union{EnergyKernel,GaussianKernel}, data::Matrix{Float64}, n::Int; kappa = nothing, max_iterations = 500, tolerance = 1e-10, n_threads = Threads.nthreads(), rng = Random.default_rng(), verbose = false, weights = nothing, target = nothing, target_weights = nothing, _n0_factor = 0.2, _subsampling = :uniform) -> (points, converged, iterations)`; `GaussianKernel{Symbol}` throws `ArgumentError("GaussianKernel bandwidth must be resolved; call resolve first")`.
  - `_mmd_trajectory(k::GaussianKernel{Float64}, data, n; max_iterations, rng, weights = nothing, target = nothing, target_weights = nothing) -> Vector{Float64}`: objective after each full-data sweep (index 1 is the initial objective).
  - `SupportPointSplitter(kernel = GaussianKernel(), kappa = 50)` constructs.

- [ ] **Step 1: Edit the tests that pin the old behavior**

In `test/test_optimizer.jl`, testset `support_points with GaussianKernel`:

1. Rename `"objective is non-increasing along accepted steps"` to `"objective is non-increasing along full-data sweeps"` (body unchanged).
2. In `"argument validation"` delete the line `@test_throws ArgumentError SPlit.support_points(k, data, 5; kappa = 10)` and add `@test_throws ArgumentError SPlit.support_points(k, data, 5; kappa = 0)`.
3. Delete the testset `"relative-decrease rule stops a flat objective honestly"` (it passes `rtol`). Replace it with:

```julia
  @testset "a flat objective stops by the displacement rule" begin
    data = randn(MersenneTwister(143), 200, 2)
    _, conv, iters = SPlit.support_points(
      GaussianKernel(1e-3),   # far below the row spacing: nothing moves
      data,
      20;
      max_iterations = 50,
      rng = MersenneTwister(144),
    )
    @test conv == true
    @test iters < 50
  end

  @testset "stochastic mode: runs, reproducible, full-data when kappa ≥ N" begin
    data = randn(MersenneTwister(145), 400, 2)
    a, _, ia = SPlit.support_points(
      k,
      data,
      20;
      kappa = 80,
      max_iterations = 60,
      rng = MersenneTwister(146),
    )
    b, _, _ = SPlit.support_points(
      k,
      data,
      20;
      kappa = 80,
      max_iterations = 60,
      rng = MersenneTwister(146),
    )
    @test a == b && size(a) == (20, 2) && 1 <= ia <= 60
    for j = 1:2
      lo, hi = extrema(view(data, :, j))
      @test all(lo .<= a[:, j] .<= hi)
    end
    full, cf, _ =
      SPlit.support_points(k, data, 20; max_iterations = 30, rng = MersenneTwister(147))
    big, cb, _ = SPlit.support_points(
      k,
      data,
      20;
      kappa = 400,
      max_iterations = 30,
      rng = MersenneTwister(147),
    )
    @test full == big && cf == cb
  end

  @testset "stochastic mode beats the initial sample under MMD" begin
    rng = MersenneTwister(148)
    data = vcat(randn(rng, 300, 2) .- 3, randn(rng, 300, 2) .+ 3)
    init = SPlit._initial_points(MersenneTwister(149), copy(data), 30, SPlit._data_bounds(data))
    pts, _, _ = SPlit.support_points(
      k,
      data,
      30;
      kappa = 100,
      max_iterations = 100,
      rng = MersenneTwister(149),
    )
    @test SPlit._mmd_objective(k, pts, data) < SPlit._mmd_objective(k, init, data)
  end
```

(`_initial_points` draws first from the rng in `support_points`, so `init` is the same starting sample as the run's.)

In `test/test_splitter.jl` replace the testset `"constructor rejects kappa with GaussianKernel"` with:

```julia
  @testset "constructor accepts kappa with GaussianKernel" begin
    s = SupportPointSplitter(kernel = GaussianKernel(), kappa = 50)
    @test s.kappa == 50
    @test s.kernel.bandwidth === :median
    @test_throws ArgumentError SupportPointSplitter(kernel = GaussianKernel(), kappa = 0)
  end

  @testset "stochastic Gaussian split runs, stores the bandwidth, beats random under MMD" begin
    rng = MersenneTwister(75)
    data = vcat(randn(rng, 600, 2) .- 2, randn(rng, 600, 2) .+ 2)
    s = SupportPointSplitter(
      kernel = GaussianKernel(),
      kappa = 200,
      max_iterations = 100,
      rng = MersenneTwister(76),
    )
    r = datasplit(s, data)
    @test r.method.kernel isa GaussianKernel{Float64}
    @test r.method.kappa == 200
    @test length(test_indices(r)) == 240
    q = splitquality(data, r; kernel = r.method.kernel)
    rand_q = Float64[]
    for i = 1:20
      perm = randperm(MersenneTwister(3_000 + i), 1200)
      fake = SPlit.SplitResult(perm[241:end], perm[1:240], true, 0, s)
      push!(rand_q, splitquality(data, fake; kernel = r.method.kernel))
    end
    @test q < mean(rand_q)
  end
```

(Check that `test/test_splitter.jl` has `using Random` and `using Statistics`; add whichever is missing at the top.)

- [ ] **Step 2: Run both files to verify the new tests fail**

Run: `julia --project=<worktree> <worktree>/test/test_optimizer.jl` and `… test/test_splitter.jl`
Expected: `ArgumentError` from the constructor / `support_points` for the `kappa` cases; the flat-objective test fails on `rtol`-free behavior only after Step 3 changes (it may already pass; that is fine).

- [ ] **Step 3: Unify the loop and delete the Armijo path**

In `src/optimizer.jl`:

1. Change the energy method's signature to `function support_points(k::Union{EnergyKernel,GaussianKernel}, data::Matrix{Float64}, n::Int; …)` (same keywords), add as the first statement `isresolved(k) || throw(ArgumentError("GaussianKernel bandwidth must be resolved; call resolve first"))`, and change the sweep call to `_mm_sweep!(k, new_points, current_const, points, sub, sub_w, running_const, alpha, bounds, n_threads)`. Nothing else in the loop changes.
2. Replace the docstring above it with the merged one:

```julia
"""
    support_points(kernel, data, n; kwargs...) -> (points, converged, iterations)

Compute `n` support points for `data` (rows are observations) under `kernel`,
by the majorization–minimization (MM) sweep of that kernel: the closed-form
update of Mak & Joseph (2018) for `EnergyKernel`, and for `GaussianKernel`
the mean-shift update in which each point moves toward the kernel-weighted
mean of the data, pushed by the linearized repulsion from the other points
(see the Methods page). Both sweeps cost one pass over the data and the
point set, and neither increases the objective on full data. Returns the
points, whether the point-movement tolerance was reached, and the number of
iterations actually used. A `GaussianKernel` must be resolved (numeric
bandwidth); `datasplit` resolves it.

Convergence compares the largest *squared* displacement of any support point
in one iteration to `tolerance`. In stochastic mode (`kappa !== nothing`),
the running-average weight for iteration `i` is `n0 / (i + n0)` with
`n0 = 0.2n`, which decays toward zero as iterations proceed, so convergence
there partly reflects this step-size decay rather than the objective
flattening out. `n0 = 0.2n` is an implementation constant, not from the
papers, chosen by a small convergence experiment (see `_n0_factor`, an
internal tuning knob not exposed on `SupportPointSplitter`).

`weights` (one non-negative entry per row, `nothing` for uniform) makes the
points approximate the weighted empirical distribution `Σ w̄ᵢ δ(xᵢ)`: the
data sums in the MM update carry `ŵᵢ = N w̄ᵢ`, which is exactly `1.0` for
uniform weights. In stochastic mode `_subsampling` (internal) selects how
the `kappa` rows are drawn: `:uniform` draws them uniformly and rescales
their weights to mean one within the subsample; `:proportional` draws them
with probability proportional to the weights and treats the subsample as
uniform (this needs at least `kappa` rows with positive weight). The
default was chosen by the weighted-`kappa` experiment on the Design
experiments page. A constant weight vector is treated as `nothing`, so
uniform weights take the unweighted path and reproduce it exactly.

`target` (a matrix with the same columns as `data`) makes the points
approximate the empirical distribution of `target` instead of `data`:
the data term of the objective runs over the rows of `target`, weighted by
`target_weights` (sum-one normalized, `nothing` for uniform; a constant
vector is treated as `nothing`), while the initial points and the bounding
box come from `data`, whose rows the points are later rounded to. In
stochastic mode `kappa` subsamples the rows of `target`. `weights` is only
for the case without a target; giving both is an `ArgumentError`. A target
with duplicate rows is jittered by 1e-3 of its column range like the data,
so weighting a reference is equivalent to duplicating its rows only up to
that jitter.
"""
```

- Delete `_armijo_step!` and `_first_step` with their comments, and delete the Gaussian `support_points` docstring and method entirely. Keep `_mmd_objective` (all methods), `_mean_kernel`, and `_mmd_gradient!`.
- Replace `_mmd_trajectory` with:

```julia
# Test helper: MMD² objective (up to its constant) after each full-data
# Gaussian MM sweep, weighted when `weights` is given, toward `target` when
# one is given. The Gaussian twin of `_objective_trajectory`.
function _mmd_trajectory(
  k::GaussianKernel{Float64},
  data::Matrix{Float64},
  n::Int;
  max_iterations::Int,
  rng::AbstractRNG,
  weights::Union{Nothing,AbstractVector} = nothing,
  target::Union{Nothing,AbstractMatrix} = nothing,
  target_weights::Union{Nothing,AbstractVector} = nothing,
)
  R, w_hat, w_bar = _resolve_target(data, weights, target, target_weights)
  bounds = _data_bounds(data)
  points = _initial_points(rng, copy(data), n, bounds)
  new_points = similar(points)
  running_const = zeros(n)
  current_const = zeros(n)
  traj = Float64[_mmd_objective(k, points, R, w_bar)]
  for _ = 1:max_iterations
    _mm_sweep!(k, new_points, current_const, points, R, w_hat, running_const, 1.0, bounds, 1)
    points, new_points = new_points, points
    push!(traj, _mmd_objective(k, points, R, w_bar))
  end
  return traj
end
```

- Update the file's header comment (line 9 mentions "gradient descent with Armijo backtracking") to say the Gaussian path is an MM sweep too.

In `src/splitter.jl`:

- Delete lines 78-79 (the `GaussianKernel && kappa` error).
- In the docstring, `kappa` bullet: drop nothing (it already reads generically); `tolerance` bullet: delete the sentence starting `For \`GaussianKernel\`, convergence never fires` through `(not exposed here).`;`kernel` bullet unchanged.

- [ ] **Step 4: Run the whole suite, then the changed files on Julia 1.12**

Run: `julia --project=<worktree> -e "using Pkg; Pkg.test()"`; then `julia +1.12 --project=<worktree> <worktree>/test/test_optimizer.jl` and `… test/test_splitter.jl`.
Expected: all pass. Tests elsewhere (`test_weights.jl`, `test_properties.jl`, `test_multiplet.jl`, `test_comparison.jl`) that use `GaussianKernel` are property tests and must pass without edits; if one fails, report which and why rather than editing it.

- [ ] **Step 5: Commit**

```bash
git add src/optimizer.jl src/splitter.jl test/test_optimizer.jl test/test_splitter.jl
git commit -m "feat: Run the Gaussian kernel through the MM loop and allow kappa

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: splitiq parity

**Files:**

- Test: `splitiq/tests/test_datasplit.py` (append)
- Modify: `docs/src/30-python.md` (the `kappa` mention near line 36/188), `splitiq/README.md` or `splitiq/docs/*.md` wherever `kappa` is described as energy-only (grep `kappa` under `splitiq/docs` and `splitiq/README.md`; if no text restricts it to the energy kernel, add one sentence to the `kappa` parameter description in `splitiq/src/splitiq/split.py` docstrings: "Works with both kernels.")

**Interfaces:**

- Consumes: Task 2's `SupportPointSplitter(kernel = GaussianKernel(), kappa = …)` through the existing wrapper (`kappa` already passes through).

- [ ] **Step 1: Write the failing test**

Append to `splitiq/tests/test_datasplit.py`:

```python
def test_gaussian_kernel_accepts_kappa() -> None:
    data = _data(600, seed=7)
    result = datasplit(
        data, ratio=0.2, kernel='gaussian', kappa=150, seed=1, max_iterations=30
    )
    assert len(result.test_indices) == 120
    assert result.bandwidth is not None and result.bandwidth > 0
```

(`_data` is the module's helper; `result.bandwidth` is the resolved bandwidth field documented in `split.py:55`.)

- [ ] **Step 2: Run it against the dev checkout**

From `<worktree>/splitiq`: `source scripts/setup_julia_dev.sh` (sets `PYTHON_JULIACALL_PROJECT`/`PYTHON_JULIACALL_EXE` to the worktree; see the memory note `splitiq-julia-dev-override`), then `uv run pytest tests/test_datasplit.py -k kappa -q`.
Expected: PASS (the Julia side no longer raises). If it raises `ValueError`/`JuliaError` mentioning `kappa`, the dev override is not pointing at the worktree — fix the environment, not the test.

- [ ] **Step 3: Docs mention**

In `docs/src/30-python.md` and the splitiq docs, make sure the `kappa` description does not say energy-only and add "(both kernels)" where the parameter is listed. Run `uv run ruff check` and `uv run ruff format --check` in `splitiq/`.

- [ ] **Step 4: Commit**

```bash
git add splitiq docs/src/30-python.md
git commit -m "test: Cover kappa with the Gaussian kernel in splitiq

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Benchmark Armijo vs MM

**Files:**

- Create: `benchmark/gaussian_update.jl`
- Create (generated): `docs/src/assets/benchmarks/gaussian_update.md`
- Modify: `docs/src/25-design-experiments.md` (append a section before `## [Compress++ cost rule]` or at the end, matching the page's style)

**Interfaces:**

- Consumes: `SPlit._mmd_gradient!(G, k, points, data, w_hat, n_threads)`, `SPlit._mmd_objective(k, points, data)`, `SPlit._data_bounds`, `SPlit._initial_points`, `SPlit.resolve(k, data, rng)`, `SPlit.select_nearest(X, points)`, `SPlit.support_points`, `mmd(X, Y, k; estimator = Exact())`, `benchmark/datasets.jl`'s `datasets(N, rng)`.
- Produces: the table file and the docs section.

- [ ] **Step 1: Write the script**

```julia
# Armijo projected gradient (the Gaussian optimizer before M6, carried here
# verbatim) versus the Gaussian MM sweep (`support_points`), on the four
# benchmark datasets at N = 1,000 and 10,000, n = 0.2N, `:median`
# bandwidth, three seeds. Both optimizers start from the same initial
# points and stop by their own rules (Armijo: displacement 1e-10 or
# relative decrease 1e-8, at least 2 iterations; MM: displacement 1e-10),
# both capped at 200 iterations at N = 1,000 and 100 at N = 10,000 as in
# `run.jl`. Per cell: wall time (min over seeds), iterations (mean), exact
# Gaussian MMD between the selected rows and the data (mean), and the same
# MMD for a uniform random subset. Writes
# `docs/src/assets/benchmarks/gaussian_update.md`. Run:
# `julia -t auto --project=benchmark benchmark/gaussian_update.jl [--quick]`.

using SPlit, Random, Statistics, LinearAlgebra

include(joinpath(@__DIR__, "datasets.jl"))

const QUICK = "--quick" in ARGS
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)
const SIZES = QUICK ? [1_000] : [1_000, 10_000]
const SEEDS = QUICK ? [0] : [0, 1, 2]
const OUTFILE = QUICK ? "gaussian_update_quick.md" : "gaussian_update.md"

# --- the pre-M6 Armijo optimizer -------------------------------------------
function armijo_step!(new_points, points, G, f0, t0, k, data, bounds)
  t = t0
  for _ = 1:30
    @inbounds for m in axes(points, 1), j in axes(points, 2)
      new_points[m, j] = clamp(points[m, j] - t * G[m, j], bounds[j, 1], bounds[j, 2])
    end
    decrease = 0.0
    @inbounds for m in axes(points, 1), j in axes(points, 2)
      decrease += G[m, j] * (points[m, j] - new_points[m, j])
    end
    f_new = SPlit._mmd_objective(k, new_points, data)
    f_new <= f0 - 1e-4 * decrease && return t, f_new
    t /= 2
  end
  return 0.0, f0
end

function first_step(G, bounds)
  n = size(G, 1)
  scale = median(view(bounds, :, 2) .- view(bounds, :, 1))
  return 0.1 * scale / max(maximum(norm(view(G, m, :)) for m = 1:n), eps())
end

function armijo_support_points(k, data, points0; max_iterations, tolerance = 1e-10, rtol = 1e-8)
  bounds = SPlit._data_bounds(data)
  n = size(points0, 1)
  points = copy(points0)
  new_points = similar(points)
  G = similar(points)
  w_hat = ones(size(data, 1))
  f = SPlit._mmd_objective(k, points, data)
  t = 1.0
  iteration = 0
  converged = false
  while !converged && iteration < max_iterations
    iteration += 1
    SPlit._mmd_gradient!(G, k, points, data, w_hat, Threads.nthreads())
    t0 = iteration == 1 ? first_step(G, bounds) : 2t
    f_prev = f
    t, f = armijo_step!(new_points, points, G, f, t0, k, data, bounds)
    t == 0.0 && break
    max_move = 0.0
    @views for m = 1:n
      max_move = max(max_move, sum(abs2, new_points[m, :] .- points[m, :]))
    end
    points, new_points = new_points, points
    rel = abs(f_prev - f) / max(abs(f), 1e-12)
    converged = iteration >= 2 && (max_move < tolerance || rel < rtol)
  end
  return points, converged, iteration
end
# ---------------------------------------------------------------------------

function cell(name, X, n, max_iter)
  N = size(X, 1)
  rows = Dict{String,Vector{Tuple{Float64,Int,Float64}}}()   # method => (time, iters, mmd)
  rand_mmd = Float64[]
  for seed in SEEDS
    k = SPlit.resolve(GaussianKernel(), X, MersenneTwister(100 + seed))
    Z = SPlit.preprocess(X)
    bounds = SPlit._data_bounds(Z)
    init = SPlit._initial_points(MersenneTwister(200 + seed), copy(Z), n, bounds)
    quality(sel) = mmd(Z[sel, :], Z, k; estimator = Exact())
    # Armijo
    t = @elapsed (pts, _, it) = armijo_support_points(k, Z, init; max_iterations = max_iter)
    push!(get!(rows, "armijo", []), (t, it, quality(SPlit.select_nearest(Z, pts))))
    # MM, full data (same initial points: same rng draw)
    t = @elapsed (pts, _, it) = SPlit.support_points(
      k, Z, n; max_iterations = max_iter, rng = MersenneTwister(200 + seed),
      n_threads = Threads.nthreads())
    push!(get!(rows, "mm", []), (t, it, quality(SPlit.select_nearest(Z, pts))))
    # MM, kappa = 1,000 at N = 10,000
    if N >= 10_000
      t = @elapsed (pts, _, it) = SPlit.support_points(
        k, Z, n; kappa = 1_000, max_iterations = max_iter,
        rng = MersenneTwister(200 + seed), n_threads = Threads.nthreads())
      push!(get!(rows, "mm kappa=1000", []), (t, it, quality(SPlit.select_nearest(Z, pts))))
    end
    push!(rand_mmd, quality(sort(randperm(MersenneTwister(300 + seed), N)[1:n])))
  end
  return rows, mean(rand_mmd)
end

io = IOBuffer()
println(io, "| dataset | N | method | time (s) | iterations | MMD selected | MMD random |")
println(io, "|---|---:|---|---:|---:|---:|---:|")
for N in SIZES
  max_iter = N >= 10_000 ? 100 : 200
  n = round(Int, 0.2N)
  for (name, X) in datasets(N, MersenneTwister(N))
    # warm-up
    cell(name, X[1:min(N, 300), :], 60, 5)
    rows, r = cell(name, X, n, max_iter)
    for method in ("armijo", "mm", "mm kappa=1000")
      haskey(rows, method) || continue
      v = rows[method]
      println(io, "| $name | $N | $method | $(round(minimum(first.(v)); digits = 2)) | ",
        "$(round(mean(getindex.(v, 2)); digits = 1)) | ",
        "$(round(mean(last.(v)); sigdigits = 3)) | $(round(r; sigdigits = 3)) |")
    end
    @info "done" name N
  end
end
write(joinpath(OUT, OUTFILE), String(take!(io)))
```

`SPlit.preprocess(X)` returns the standardized `Matrix{Float64}` (`src/preprocessing.jl:295`), the same matrix `datasplit` optimizes on.

- [ ] **Step 2: Quick run**

Run: `julia -t auto --project=<worktree> <worktree>/benchmark/gaussian_update.jl --quick` (the `benchmark` project may not be instantiated; the main project has every dependency this script needs).
Expected: `docs/src/assets/benchmarks/gaussian_update_quick.md` with 8 rows; delete it afterwards (`git rm`-free, it is untracked).

- [ ] **Step 3: Full run**

Run: `julia -t auto --project=<worktree> <worktree>/benchmark/gaussian_update.jl` (expect a few minutes; Armijo at N = 10,000 is the slow part).
Expected: `docs/src/assets/benchmarks/gaussian_update.md` with 20 rows.

- [ ] **Step 4: Design experiments section**

Append to `docs/src/25-design-experiments.md`, in the page's style (claim, numbers, table link, reproduce command):

````markdown
## [Gaussian update rule](@id gaussian-update)

`support_points(::GaussianKernel, …)` minimizes MMD² by an MM sweep (the
mean-shift majorizer of the data term plus the L-smooth majorizer of the
repulsion; see [Methods](@ref methods)) instead of the projected gradient
with Armijo backtracking it used before. A sweep is one pass over the data
and the point set; an Armijo iteration evaluates the objective up to 30
times. Measured on the four benchmark datasets at N = 1,000 and 10,000,
n = 0.2N, `:median` bandwidth, three seeds: <fill in from the table: the
time ratio range, whether the MMD of the selected rows is within X% of the
Armijo rows, and the `kappa = 1,000` row's time and MMD>. The damped
uniform-weight fixed point of Belhadji, Sharp & Marzouk (2025, eq. 29)
was also tried and diverged on every dataset, because its denominator
crosses zero where the point set fits the data. Full table:
[`assets/benchmarks/gaussian_update.md`](assets/benchmarks/gaussian_update.md).
Reproduce with:

```sh
julia -t auto --project=benchmark benchmark/gaussian_update.jl
```

````

Replace the `<fill in …>` sentence with the measured numbers (that placeholder is the only text you may not leave in place).

- [ ] **Step 5: Commit**

```bash
git add benchmark/gaussian_update.jl docs/src/assets/benchmarks/gaussian_update.md docs/src/25-design-experiments.md
git commit -m "perf: Benchmark the Gaussian MM sweep against the Armijo path

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Re-derive the Benchmarks page

**Files:**

- Modify: `benchmark/rounding.jl:50-92` (`gaussian_support_points_from`) and its two call sites (`rtol` keyword)
- Regenerate: `docs/src/assets/benchmarks/results.md`, `quality.png`, `time.png`, `selection.png` (by `benchmark/run.jl`), `rounding.md`, `rounding.png` (by `benchmark/rounding.jl`)
- Modify: `docs/src/20-benchmarks.md` where its numbers or claims about `support points · gaussian` change

**Interfaces:**

- Consumes: Task 1's `SPlit._mm_sweep!(k, …)`.

- [ ] **Step 1: Switch the rounding script's local loop to the sweep**

Replace `gaussian_support_points_from` with:

```julia
# Local copy of the Gaussian support-point loop (`support_points` in
# `src/optimizer.jl`, full-data MM sweep), starting from a given initial
# point set instead of `_initial_points`.
function gaussian_support_points_from(
  k::GaussianKernel{Float64},
  data::Matrix{Float64},
  points0::Matrix{Float64};
  max_iterations::Int,
  tolerance::Float64,
  n_threads::Int,
)
  bounds = SPlit._data_bounds(data)
  n = size(points0, 1)
  points = copy(points0)
  new_points = similar(points)
  running_const = zeros(n)
  current_const = zeros(n)
  w_hat = ones(size(data, 1))
  iteration = 0
  converged = false
  while !converged && iteration < max_iterations
    iteration += 1
    SPlit._mm_sweep!(k, new_points, current_const, points, data, w_hat, running_const, 1.0, bounds, n_threads)
    max_move = 0.0
    @views for m = 1:n
      max_move = max(max_move, sum(abs2, new_points[m, :] .- points[m, :]))
    end
    points, new_points = new_points, points
    converged = max_move < tolerance
  end
  return points, converged, iteration
end
```

and delete `rtol = 1e-8,` at its two call sites. Update the header comment if it mentions Armijo.

- [ ] **Step 2: Run both benchmark scripts**

Run: `julia -t auto --project=<worktree> <worktree>/benchmark/run.jl` then `… benchmark/rounding.jl`. If the `benchmark` project is needed for `CairoMakie`/`Distributions`, instantiate it once: `julia --project=<worktree>/benchmark -e 'using Pkg; Pkg.develop(path = "<worktree>"); Pkg.instantiate()'` (its `Manifest.toml` is git-ignored).
Expected: the six asset files change; only `support points · gaussian` rows should move materially. Keep the previous `results.md` (`git show HEAD:docs/src/assets/benchmarks/results.md`) to compare.

- [ ] **Step 3: Update the Benchmarks page**

In `docs/src/20-benchmarks.md`, re-read sections 1-3 and "How it was run" against the new `results.md`/`rounding.md` and change every number or claim about `support points · gaussian` (speed ratio in section 2, the "Rows kept from the initial sample" and "median move" figures in section 3, the `max_iterations` note in the methods table if it changed). Claims about the other methods stay unless their numbers moved beyond rounding (they should not; report if they did).

- [ ] **Step 4: Commit**

```bash
git add benchmark/rounding.jl docs/src/assets/benchmarks docs/src/20-benchmarks.md
git commit -m "docs: Re-derive the benchmarks for the Gaussian MM sweep

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Docs, roadmap, README, AGENTS.md

**Files:**

- Modify: `docs/src/10-methods.md:122-178` (the Gaussian section), `docs/src/85-roadmap.md` (current-state row for `GaussianKernel`, M6 section, changelog), `README.md:192-196`, `AGENTS.md` (the `GaussianKernel has no kappa mode` gotcha and the `Gaussian optimizer: the first trial step…` gotcha)

- [ ] **Step 1: Methods page**

Replace the text from `Support points under it minimize, up to a constant,` through `The stochastic \`kappa\` mode is not available for this kernel.` with:

````markdown
Support points under it minimize, up to a constant,

```math
f(\xi) = \frac{1}{n^2} \sum_{i,j} k(\xi_i, \xi_j)
  - \frac{2}{nN} \sum_{i,l} k(\xi_i, x_l).
```

The optimizer (`support_points(::GaussianKernel, …)`) is a
majorization–minimization sweep of the same shape as the energy sweep.
The data term ``-k(\xi, x)`` is concave in ``\|\xi - x\|^2``, so its tangent
at the current point is an upper bound whose minimizer is the mean-shift
step (Fukunaga & Hostetler, 1975): the kernel-weighted mean of the data. The
repulsion ``k(\xi_i, \xi_j)`` is bounded above by its quadratic
``L``-smooth majorizer with ``L = 2e^{-3/2}/\sigma^2``, the largest
Hessian eigenvalue of a Gaussian, split evenly over the two points. Per
point, with ``s_0 = \sum_l k(\xi_m, x_l)``, ``s_1 = \sum_l k(\xi_m, x_l)\,x_l``,
``r_0 = \sum_{j \ne m} k(\xi_m, \xi_j)``, ``r_1 = \sum_{j \ne m} k(\xi_m, \xi_j)\,\xi_j``,

```math
A = \frac{s_0}{N\sigma^2}, \quad
\mathrm{ms} = \frac{s_1}{s_0}, \quad
\mathrm{rep} = \frac{r_0\,\xi_m - r_1}{n\sigma^2}, \quad
B = \frac{4(n-1)e^{-3/2}}{n\sigma^2}, \qquad
\xi_m \leftarrow \operatorname{clamp}\!\left(\frac{A\,\mathrm{ms} + B\,\xi_m + \mathrm{rep}}{A + B}\right).
```

``A + B > 0`` always, the sweep is one pass over the data and the point
set, and on full data the objective never increases (the majorizer is
tangent at the current points, so its minimizer cannot be worse). The
attraction weight ``A/(A+B)`` is at most about ``0.53``, so a sweep moves a
point at most about halfway to its mean-shift target; per iteration this
descends less than a line search would, but an iteration costs a single
pass, which is what makes the stochastic mode affordable. With
``\kappa < N`` the sweep runs on a fresh subsample each iteration with the
running-average blend of the energy path (``n_0 = 0.2n``); `weights` enter
the data sums as ``\hat w``, and a reference replaces the data rows, exactly
as for `EnergyKernel`. Convergence is the displacement rule. The design
record (`docs/superpowers/specs/2026-09-04-gaussian-mm-update-design.md`)
explains why the weighted mean-shift map of Belhadji, Sharp & Marzouk
(2025) is not used as written: it optimizes the subset's weights too, and
the package's selected subset is uniform.

When `bandwidth = :median`, ``\sigma`` is the median pairwise distance over
(a sample of) the standardized rows (Gretton et al., 2012), resolved once
per `datasplit` and stored in `result.method.kernel`. A bandwidth far below
the row spacing makes the objective flat: the points barely move and the
displacement rule stops at the initial sample.
````

Also change the first sentence of the section (`… and \`support_points\` switches from the MM update to projected gradient descent.`) to`… and \`support_points\` runs the Gaussian MM sweep described below.` Add the two new references (Fukunaga & Hostetler 1975; Fashing & Tomasi 2005) to the page's reference list if it has one, in its style.

- [ ] **Step 2: Roadmap**

- Current-state row for `SupportPointSplitter` with `GaussianKernel`: `Minimizes squared MMD by a mean-shift MM sweep (roadmap M6); \`kappa\` works as for \`EnergyKernel\`. A \`:median\` bandwidth is resolved at \`datasplit\` time and the resolved kernel is stored in \`result.method.kernel\`.`
- Replace the M6 section body with `Done (2026-09-04).` plus three sentences: what landed (MM sweep, `kappa`), the measured outcome from Task 4's table (time ratio, MMD parity), and that the paper's weighted map was not adopted (uniform subset). Keep the heading `### M6: MMD gradient-flow update` (drop "(exploratory)").
- Changelog line (find the list of dated lines near the top or bottom of the page; keep its format): `- 2026-09-04: M6 (Gaussian MM sweep, \`kappa\` for \`GaussianKernel\`) done.`

- [ ] **Step 3: README and AGENTS.md**

`README.md:192-196`: the `GaussianKernel` bullet becomes: support points minimize the squared MMD by a mean-shift MM sweep instead of the energy-distance MM step; `kappa` works as for `EnergyKernel`; the resolved bandwidth is stored in `result.method.kernel`.

`AGENTS.md`: replace `- \`GaussianKernel\` has no \`kappa\` mode; its \`:median\` bandwidth is resolved …` with `- \`GaussianKernel\`'s \`:median\` bandwidth is resolved at \`datasplit\` time and the resolved kernel is stored in \`result.method.kernel\`; \`kappa\` works as for \`EnergyKernel\`.` Replace the `- Gaussian optimizer: the first trial step …` gotcha with: `- Gaussian optimizer: an MM sweep (mean-shift data term, L-smooth majorized repulsion with \`B = 4(n−1)e^{−3/2}/(nσ²)\`), sharing the energy loop, \`kappa\` blend, and displacement rule; \`alpha = 1\` is the pure MM step and the descent test enforces monotonicity. There is no line search and no \`rtol\`.`

- [ ] **Step 4: Build the docs and lint**

Run: `julia --project=<worktree>/docs -e 'using Pkg; Pkg.develop(path = "<worktree>"); Pkg.instantiate(); include("<worktree>/docs/make.jl")'` (skip deploy; a local build only) and `pre-commit run --all-files` from the worktree.
Expected: no Documenter cross-reference errors (`@ref methods`, `@ref gaussian-update`), lint clean.

- [ ] **Step 5: Commit**

```bash
git add docs/src/10-methods.md docs/src/85-roadmap.md README.md AGENTS.md
git commit -m "docs: Describe the Gaussian MM sweep and close roadmap M6

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Amendment after Task 4 (user decision, option 1)

The spec's Amendment section is binding from here: Armijo stays on full data, the MM sweep runs in stochastic mode only. Task 5 shrinks to a signature fix (no benchmark reruns); Task 4's script and docs section are revised in Task 8; Task 6's docs text must describe both paths.

### Task 7: Armijo on full data, MM sweep in stochastic mode

**Files:**

- Modify: `src/optimizer.jl`, `src/splitter.jl` (docstring only), `test/test_optimizer.jl`
- Reference: `git show c9e2809:src/optimizer.jl` (the pre-M6 file; lines 421-475 hold `_armijo_step!` and `_first_step`, 472-577 the Gaussian `support_points` docstring and method, 579-606 the Armijo `_mmd_trajectory`) and `git show c9e2809:test/test_optimizer.jl`

**Interfaces:**

- Produces:
  - `_support_points_mm(k::Union{EnergyKernel,GaussianKernel{Float64}}, data::Matrix{Float64}, n::Int; kappa, max_iterations, tolerance, n_threads, rng, verbose, weights, target, target_weights, _n0_factor, _subsampling) -> (points, converged, iterations)` — the current unified loop, renamed (body unchanged).
  - `support_points(k::EnergyKernel, data, n; kappa = nothing, max_iterations = 500, tolerance = 1e-10, n_threads = Threads.nthreads(), rng = Random.default_rng(), verbose = false, weights = nothing, target = nothing, target_weights = nothing, _n0_factor = 0.2, _subsampling = :uniform)` — forwards every keyword to `_support_points_mm`; the current merged docstring moves above it, with its Gaussian sentences removed.
  - `support_points(k::GaussianKernel, data, n; kappa = nothing, max_iterations = 500, tolerance = 1e-10, rtol =
    1e-8, n_threads, rng, verbose, weights, target, target_weights, _n0_factor = 0.2, _subsampling = :uniform)`:
    `isresolved(k)` check; `M = target === nothing ? size(data, 1) : size(target, 1)`; `stochastic = kappa !==
    nothing && kappa < M`; when stochastic, `return _support_points_mm(k, data, n; kappa, max_iterations,
    tolerance, n_threads, rng, verbose, weights, target, target_weights, _n0_factor, _subsampling)`; otherwise the
    pre-M6 Armijo body restored verbatim (from `git show c9e2809:src/optimizer.jl` lines 512-577) minus its
    `kappa` `ArgumentError` line (kappa validation `kappa === nothing || kappa > 0 || throw(ArgumentError("kappa
    must be positive, got $kappa"))` replaces it, before the `stochastic` test). Its docstring is the pre-M6 one
    (lines 472-511) rewritten so the first paragraph says: full data → projected gradient with Armijo backtracking
    (unchanged text), stochastic mode (`kappa` below the number of target rows) → the Gaussian MM sweep (mean-
    shift data term, majorized repulsion, see the Methods page) with the energy path's running-average blend and
    the displacement rule only; `rtol` applies to the full-data path only. Keep the weights/target paragraphs.
  - `_armijo_step!`, `_first_step` restored verbatim (lines 421-471) directly above the Gaussian method.
  - `_mmd_trajectory(k::GaussianKernel{Float64}, data, n; max_iterations, rng, weights, target, target_weights)` restored verbatim as the Armijo trajectory (lines 579-606); the current sweep-based body is deleted (the one-sweep tests in "Gaussian MM sweep" cover the sweep).
  - Method order in the file: `_mm_sweep!` (energy), the two Task-1 methods, `_resolve_target`, `_draw_subsample`, docstring + `support_points(::EnergyKernel)`, `_support_points_mm`, `_objective_trajectory`, `_mmd_objective`s, `_mmd_gradient!`, `_armijo_step!`, `_first_step`, docstring + `support_points(::GaussianKernel)`, `_mmd_trajectory`.

- [ ] **Step 1: Test edits (RED first where applicable)**

In `test/test_optimizer.jl`, testset `support_points with GaussianKernel`:

- Rename `"objective is non-increasing along full-data sweeps"` back to `"objective is non-increasing along accepted steps"` (body unchanged; it uses `_mmd_trajectory`, now Armijo again).
- Restore the pre-M6 testset `"relative-decrease rule stops a flat objective honestly"` verbatim from `git show c9e2809:test/test_optimizer.jl` (it passes `rtol = 1e-3` and asserts `conv && 2 <= iters < 300`).
- Change `"a flat objective stops by the displacement rule"` into the stochastic version: add `kappa = 100` to its `support_points` call (N = 200, so stochastic) and rename it `"a flat objective stops by the displacement rule in stochastic mode"`; assertions unchanged (`conv == true`, `iters < 50`).
- Keep `"stochastic mode: runs, reproducible, full-data when kappa ≥ N"` and `"stochastic mode beats the initial sample under MMD"` unchanged.
- In `"concentrated weights pull support points toward the heavy cluster"` set both `max_iterations` back to `100` (the Armijo path passes there, as before Task 2).
- Add, at the end of that testset:

```julia
  @testset "stochastic mode is the MM sweep: kappa < N never calls the line search" begin
    data = randn(MersenneTwister(150), 300, 2)
    # An unresolvable line search would report converged = false at iteration 1;
    # the MM sweep always produces a step, so a 5-iteration run reports 5.
    _, conv, iters = SPlit.support_points(
      k,
      data,
      20;
      kappa = 50,
      max_iterations = 5,
      tolerance = 1e-30,
      rng = MersenneTwister(151),
    )
    @test conv == false && iters == 5
  end
```

Run `test/test_optimizer.jl`: the restored `rtol` test errors (`rtol` is not a keyword yet) — RED.

- [ ] **Step 2: Source changes** as in Interfaces. Then in `src/splitter.jl`'s `SupportPointSplitter` docstring, `tolerance` bullet, re-add after "…rather than the objective flattening out.": `For \`GaussianKernel\` on full data, convergence never fires before the second iteration, and also triggers when the relative objective decrease falls below an internal \`rtol = 1e-8\` (not exposed here); in stochastic mode the Gaussian kernel uses the MM sweep and the displacement rule only.`

- [ ] **Step 3: Verify.** `test/test_optimizer.jl` and `test/test_splitter.jl` on Julia 1.10 and 1.12, then the full suite. `grep -n "rtol\|_armijo_step!\|_first_step\|_support_points_mm" src/optimizer.jl` shows the restored names; `grep -rn "_mm_trajectory" src test` is empty.

- [ ] **Step 4: Commit** `feat: Keep Armijo on full data and run the MM sweep in stochastic mode`.

### Task 8: Revise the benchmark script and the Design experiments section

**Files:** `benchmark/gaussian_update.jl`, `docs/src/assets/benchmarks/gaussian_update.md` (regenerated), `docs/src/25-design-experiments.md` (the `gaussian-update` section)

- Delete the script's private Armijo copy; the `armijo` arm becomes `SPlit.support_points(k, Z, n; max_iterations = max_iter, rng = MersenneTwister(200 + seed), n_threads = Threads.nthreads())` (full data → Armijo). The `mm` arm becomes a private loop: `max_iter` calls of `SPlit._mm_sweep!(k, new_points, current_const, points, Z, ones(N), running_const, 1.0, bounds, Threads.nthreads())` from `init` (with `points, new_points = new_points, points` after each), reporting `max_iter` as its iteration count. The `mm kappa=1000` arm stays `SPlit.support_points(…; kappa = 1_000, …)`. Update the header comment to say what each arm is and that the full-data sweep is not an API path.
- Rerun quick, then full (background; wait with a loop that does NOT match itself — e.g. `until grep -q "gaussian_update: done" LOG; do sleep 30; done`).
- Rewrite the docs section: it now records the decision — Armijo on full data, MM sweep in `kappa` mode — with the measured reasons (the `uniform-5d` MMD numbers, the early-stop/wall-time observation, the `kappa` speedups), the rejected over-relaxation experiment in one sentence, the table link, and the reproduce command. Commit `perf: Record the Gaussian optimizer decision with the revised benchmark`.

### Task 5 (amended): `benchmark/rounding.jl` signature fix only

Change `SPlit._mmd_gradient!(G, k, points, data, n_threads)` to `SPlit._mmd_gradient!(G, k, points, data, ones(size(data, 1)), n_threads)`; keep its Armijo loop and `rtol`. Do not rerun `run.jl` or `rounding.jl`. Commit `fix: Match the rounding benchmark to the gradient signature`. May be batched with Task 3.

### Task 6 (amended)

The Methods page keeps the Armijo description for full data and adds the MM sweep paragraph for stochastic mode (the same formulas as written in Task 6 Step 1, introduced by "With `kappa` below the number of rows, the Gaussian kernel switches to a majorization–minimization sweep of the energy sweep's shape…"), ending with the measured reason the sweep is not used on full data (link to the Design experiments section). Roadmap M6 "Done (2026-09-04)": what landed (MM sweep, `kappa` for `GaussianKernel`), the outcome (full data stays Armijo, measured), the paper's weighted map not adopted. README/AGENTS.md: `GaussianKernel` gotcha becomes "`kappa` runs the MM sweep (mean-shift data term, majorized repulsion, displacement rule); full data stays the Armijo path with its first-trial-step and `rtol` rules (measured, see Design experiments)". Keep the existing "Gaussian optimizer: the first trial step…" gotcha.
