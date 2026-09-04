# Three arms, on the four benchmark datasets at N = 1,000 and 10,000,
# n = 0.2N, `:median` bandwidth, three seeds:
#   - `armijo`: the package's full-data Gaussian path
#     (`SPlit.support_points(::GaussianKernel, …)` with `kappa = nothing`) —
#     projected gradient descent with Armijo backtracking, stopped by its
#     own rules (displacement 1e-10 or relative decrease 1e-8, at least 2
#     iterations).
#   - `mm`: the full-data MM sweep (mean-shift data term, majorized
#     repulsion), run here as a private loop over `SPlit._mm_sweep!` for a
#     fixed number of iterations — it is not reachable through the public
#     API on full data any more (see the Design experiments page for why).
#   - `mm kappa=1000`: the package's stochastic path
#     (`SPlit.support_points(::GaussianKernel, …; kappa = 1_000)`), MM sweep
#     with running-average blending, at N = 10,000 only.
# All arms start from the same initial points (drawn from the same rng
# seed) and are capped at 200 iterations at N = 1,000, 100 at N = 10,000,
# as in `run.jl`. Per cell: wall time (min over seeds), iterations (mean),
# exact Gaussian MMD between the selected rows and the data (mean), and the
# same MMD for a uniform random subset. Writes
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

# Full-data MM sweep run for a fixed number of iterations: not an API path
# (the public `support_points(::GaussianKernel, …)` runs Armijo on full
# data), kept here only to measure the sweep's own cost and quality.
function mm_support_points(k, Z, points0; max_iterations)
  n = size(points0, 1)
  points = copy(points0)
  new_points = similar(points)
  current_const = zeros(n)
  running_const = zeros(n)
  w_hat = ones(size(Z, 1))
  bounds = SPlit._data_bounds(Z)
  for _ = 1:max_iterations
    SPlit._mm_sweep!(
      k,
      new_points,
      current_const,
      points,
      Z,
      w_hat,
      running_const,
      1.0,
      bounds,
      Threads.nthreads(),
    )
    points, new_points = new_points, points
  end
  return points, false, max_iterations
end

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
    # Armijo (public full-data path; draws its own initial points from the
    # same rng seed as `init`, so the two arms share initial points)
    t = @elapsed (pts, _, it) = SPlit.support_points(
      k,
      Z,
      n;
      max_iterations = max_iter,
      rng = MersenneTwister(200 + seed),
      n_threads = Threads.nthreads(),
    )
    push!(get!(rows, "armijo", []), (t, it, quality(SPlit.select_nearest(Z, pts))))
    # MM, full data (private loop; same initial points: same rng draw)
    t = @elapsed (pts, _, it) = mm_support_points(k, Z, init; max_iterations = max_iter)
    push!(get!(rows, "mm", []), (t, it, quality(SPlit.select_nearest(Z, pts))))
    # MM, kappa = 1,000 at N = 10,000
    if N >= 10_000
      t = @elapsed (pts, _, it) = SPlit.support_points(
        k,
        Z,
        n;
        kappa = 1_000,
        max_iterations = max_iter,
        rng = MersenneTwister(200 + seed),
        n_threads = Threads.nthreads(),
      )
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
      println(
        io,
        "| $name | $N | $method | $(round(minimum(first.(v)); digits = 2)) | ",
        "$(round(mean(getindex.(v, 2)); digits = 1)) | ",
        "$(round(mean(last.(v)); sigdigits = 3)) | $(round(r; sigdigits = 3)) |",
      )
    end
    @info "done" name N
  end
end
write(joinpath(OUT, OUTFILE), String(take!(io)))
println("gaussian_update: done")
