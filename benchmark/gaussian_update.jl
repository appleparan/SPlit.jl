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

function armijo_support_points(
  k,
  data,
  points0;
  max_iterations,
  tolerance = 1e-10,
  rtol = 1e-8,
)
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
      k,
      Z,
      n;
      max_iterations = max_iter,
      rng = MersenneTwister(200 + seed),
      n_threads = Threads.nthreads(),
    )
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
