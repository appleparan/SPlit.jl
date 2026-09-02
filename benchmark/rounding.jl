# Rounding-step experiment: `SupportPointSplitter` optimizes continuous
# support points and then maps each one to its nearest unclaimed data row
# (`select_nearest`, sequential nearest-neighbor, Joseph & Vakayil 2021).
# Points are initialized at a random sample of data rows, jittered by 0.1%
# of the per-dimension range (`_initial_points`). This script measures, per
# (dataset, N), how far the optimizer actually moves the points relative to
# the spacing between data rows, and whether that leaves the rounded
# selection identical to the initial sample. It also checks two alternative
# initializations (uniform in the bounding box, and heavy jitter around the
# initial sample) to see whether starting away from data rows changes the
# outcome, and reproduces `run.jl`'s own `datasplit` path for
# `support points · gaussian` — `datasplit` resolves the `:median` bandwidth
# from the splitter's `rng` before `_initial_points` draws from it, so its
# initial sample is a different draw than the fresh-`rng` one used elsewhere
# in this script. Writes `docs/src/assets/benchmarks/rounding.md` and
# `docs/src/assets/benchmarks/rounding.png` (fraction of the initial
# sample's rows kept by `support points · gaussian` and `support points ·
# energy`, one panel per N). Run:
# `julia -t auto --project=benchmark benchmark/rounding.jl`.

using SPlit, DataFrames, Distributions, Random, Statistics, CairoMakie

include(joinpath(@__DIR__, "datasets.jl"))

const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

norm2(v) = sqrt(sum(abs2, v))

# Median nearest-neighbor distance among data rows: kNN with k = 2 (self +
# nearest other row) over the first 500 rows.
function median_nn_spacing(X::Matrix{Float64})
  tree = SPlit.NearestNeighbors.KDTree(permutedims(X))
  m = min(500, size(X, 1))
  _, dists = SPlit.NearestNeighbors.knn(tree, permutedims(X[1:m, :]), 2, true)
  return median(last(d) for d in dists)
end

# Mean test-vs-train MMD and energy distance over 5 random splits.
function random_split_scores(X::Matrix{Float64}, gk, n::Int, N::Int)
  mmds = Float64[]
  eds = Float64[]
  for i = 1:5
    test = randperm(MersenneTwister(100 + i), N)[1:n]
    rest = setdiff(1:N, test)
    push!(mmds, SPlit._exact_mmd(gk, X[test, :], X[rest, :]))
    push!(eds, SPlit._exact_energydistance(X[test, :], X[rest, :]))
  end
  return mean(mmds), mean(eds)
end

# Local copy of the Gaussian support-point loop
# (`support_points(::GaussianKernel, …)` in `src/optimizer.jl`), starting
# from a given initial point set instead of `_initial_points`.
function gaussian_support_points_from(
  k::GaussianKernel{Float64},
  data::Matrix{Float64},
  points0::Matrix{Float64};
  max_iterations::Int,
  tolerance::Float64,
  rtol::Float64,
  n_threads::Int,
)
  bounds = SPlit._data_bounds(data)
  n = size(points0, 1)
  points = copy(points0)
  new_points = similar(points)
  G = similar(points)
  f = SPlit._mmd_objective(k, points, data)
  t = 1.0

  iteration = 0
  converged = false
  while !converged && iteration < max_iterations
    iteration += 1
    SPlit._mmd_gradient!(G, k, points, data, n_threads)
    t0 = iteration == 1 ? SPlit._first_step(G, bounds) : 2t
    f_prev = f
    t, f = SPlit._armijo_step!(new_points, points, G, f, t0, k, data, bounds)
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

# Points drawn uniformly in the data's bounding box.
function uniform_box_points(rng::AbstractRNG, bounds::Matrix{Float64}, n::Int)
  p = size(bounds, 1)
  pts = Matrix{Float64}(undef, n, p)
  for j = 1:p
    lo, hi = bounds[j, 1], bounds[j, 2]
    @views pts[:, j] .= lo .+ (hi - lo) .* rand(rng, n)
  end
  return pts
end

# Initial-sample data rows plus uniform noise of ±0.5 × per-dimension range,
# clamped to the bounding box.
function heavy_jitter_points(
  rng::AbstractRNG,
  X::Matrix{Float64},
  init_rows::Vector{Int},
  bounds::Matrix{Float64},
)
  pts = X[init_rows, :]
  p = size(pts, 2)
  for j = 1:p
    lo, hi = bounds[j, 1], bounds[j, 2]
    halfrange = 0.5 * (hi - lo)
    @views pts[:, j] .=
      clamp.(pts[:, j] .+ halfrange .* (2 .* rand(rng, size(pts, 1)) .- 1), lo, hi)
  end
  return pts
end

median_displacement(pts::Matrix{Float64}, pts0::Matrix{Float64}) =
  median(norm2(view(pts, m, :) .- view(pts0, m, :)) for m = 1:size(pts, 1))

rows_kept(sel::Vector{Int}, init_rows::Vector{Int}) = length(intersect(sel, init_rows))

# Dash-style row for a baseline that has no iterations/displacement/rows-kept
# of its own (random splits, an initial sample).
function push_baseline_row!(rows, dataset, N, method, mmd::Float64, ed::Float64)
  push!(
    rows,
    (
      dataset = dataset,
      N = N,
      method = method,
      iterations = "–",
      median_move = "–",
      continuous_mmd = "–",
      rows_kept = "–",
      test_mmd = mmd,
      test_ed = ed,
    ),
  )
  return nothing
end

function process_dataset(name::String, data::Matrix{Float64}, N::Int, rows, spacings)
  X = SPlit.preprocess(data)
  gk = SPlit.resolve(GaussianKernel(), X, MersenneTwister(7))
  n = round(Int, 0.2N)
  bounds = SPlit._data_bounds(X)

  init_pts = SPlit._initial_points(MersenneTwister(1), X, n, bounds)
  init_rows = SPlit.select_nearest(X, init_pts)

  push!(spacings, (dataset = name, N = N, nn_spacing = median_nn_spacing(X)))

  rand_mmd, rand_ed = random_split_scores(X, gk, n, N)
  push_baseline_row!(rows, name, N, "random (5 seeds)", rand_mmd, rand_ed)

  init_rest = setdiff(1:N, init_rows)
  init_mmd = SPlit._exact_mmd(gk, X[init_rows, :], X[init_rest, :])
  init_ed = SPlit._exact_energydistance(X[init_rows, :], X[init_rest, :])
  push_baseline_row!(rows, name, N, "initial sample", init_mmd, init_ed)

  max_iter = N >= 10_000 ? 100 : 200

  # `run.jl`'s own `datasplit` path for `support points · gaussian`:
  # `datasplit` resolves the `:median` bandwidth from the splitter's `rng`
  # (which, for N above `MEDIAN_HEURISTIC_ROWS`, samples rows with that same
  # `rng`) *before* `support_points` draws `_initial_points` from it, so the
  # initial sample this path actually starts from is a different draw than
  # `init_rows` above.
  ds_rng = MersenneTwister(1)
  SPlit.resolve(GaussianKernel(), X, ds_rng)
  init_ds = SPlit.select_nearest(X, SPlit._initial_points(ds_rng, X, n, bounds))
  init_ds_rest = setdiff(1:N, init_ds)
  init_ds_mmd = SPlit._exact_mmd(gk, X[init_ds, :], X[init_ds_rest, :])
  init_ds_ed = SPlit._exact_energydistance(X[init_ds, :], X[init_ds_rest, :])
  push_baseline_row!(
    rows,
    name,
    N,
    "initial sample (datasplit path)",
    init_ds_mmd,
    init_ds_ed,
  )

  ds_splitter = SupportPointSplitter(
    kernel = GaussianKernel(),
    max_iterations = max_iter,
    rng = MersenneTwister(1),
  )
  r_ds = datasplit(ds_splitter, data)
  sel_ds = SPlit.test_indices(r_ds)
  rest_ds = SPlit.train_indices(r_ds)
  ds_mmd = SPlit._exact_mmd(gk, X[sel_ds, :], X[rest_ds, :])
  ds_ed = SPlit._exact_energydistance(X[sel_ds, :], X[rest_ds, :])
  kept_ds = rows_kept(sel_ds, init_ds)
  push!(
    rows,
    (
      dataset = name,
      N = N,
      method = "support points · gaussian (datasplit path)",
      iterations = r_ds.converged ? "$(r_ds.iterations)" : "$(r_ds.iterations)*",
      median_move = "–",
      continuous_mmd = "–",
      rows_kept = "$kept_ds/$(length(sel_ds))",
      test_mmd = ds_mmd,
      test_ed = ds_ed,
    ),
  )

  # (1) support points · gaussian
  pts_g, conv_g, it_g =
    SPlit.support_points(gk, X, n; max_iterations = max_iter, rng = MersenneTwister(1))
  sel_g = SPlit.select_nearest(X, pts_g)
  push_optimizer_row!(
    rows,
    name,
    N,
    "support points · gaussian",
    it_g,
    conv_g,
    pts_g,
    init_pts,
    gk,
    X,
    sel_g,
    init_rows,
  )

  # (2) support points · energy
  kappa = N >= 10_000 ? 1_000 : nothing
  pts_e, conv_e, it_e =
    SPlit.support_points(EnergyKernel(), X, n; kappa, rng = MersenneTwister(1))
  sel_e = SPlit.select_nearest(X, pts_e)
  push_optimizer_row!(
    rows,
    name,
    N,
    "support points · energy",
    it_e,
    conv_e,
    pts_e,
    init_pts,
    gk,
    X,
    sel_e,
    init_rows,
  )

  # (2b) support points · energy, full data — only at N >= 10,000, where (2)
  # above runs stochastic (kappa = 1_000); at N = 1,000, (2) is already
  # full-data, so this would duplicate it.
  if N >= 10_000
    pts_ef, conv_ef, it_ef = SPlit.support_points(
      EnergyKernel(),
      X,
      n;
      kappa = nothing,
      max_iterations = 100,
      rng = MersenneTwister(1),
    )
    sel_ef = SPlit.select_nearest(X, pts_ef)
    push_optimizer_row!(
      rows,
      name,
      N,
      "support points · energy, full data",
      it_ef,
      conv_ef,
      pts_ef,
      init_pts,
      gk,
      X,
      sel_ef,
      init_rows,
    )
  end

  # (3) gaussian, uniform-box init
  pts_ub = uniform_box_points(MersenneTwister(1), bounds, n)
  res_ub, conv_ub, it_ub = gaussian_support_points_from(
    gk,
    X,
    pts_ub;
    max_iterations = max_iter,
    tolerance = 1e-10,
    rtol = 1e-8,
    n_threads = Threads.nthreads(),
  )
  sel_ub = SPlit.select_nearest(X, res_ub)
  push_optimizer_row!(
    rows,
    name,
    N,
    "gaussian, uniform-box init",
    it_ub,
    conv_ub,
    res_ub,
    pts_ub,
    gk,
    X,
    sel_ub,
    init_rows,
  )

  # (4) gaussian, heavy-jitter init
  pts_hj = heavy_jitter_points(MersenneTwister(1), X, init_rows, bounds)
  res_hj, conv_hj, it_hj = gaussian_support_points_from(
    gk,
    X,
    pts_hj;
    max_iterations = max_iter,
    tolerance = 1e-10,
    rtol = 1e-8,
    n_threads = Threads.nthreads(),
  )
  sel_hj = SPlit.select_nearest(X, res_hj)
  push_optimizer_row!(
    rows,
    name,
    N,
    "gaussian, heavy-jitter init",
    it_hj,
    conv_hj,
    res_hj,
    pts_hj,
    gk,
    X,
    sel_hj,
    init_rows,
  )

  return nothing
end

function push_optimizer_row!(
  rows,
  dataset,
  N,
  method,
  iterations,
  converged,
  pts,
  pts0,
  gk,
  X,
  sel,
  init_rows,
)
  rest = setdiff(1:N, sel)
  test_mmd = SPlit._exact_mmd(gk, X[sel, :], X[rest, :])
  test_ed = SPlit._exact_energydistance(X[sel, :], X[rest, :])
  kept = rows_kept(sel, init_rows)
  push!(
    rows,
    (
      dataset = dataset,
      N = N,
      method = method,
      iterations = converged ? "$iterations" : "$iterations*",
      median_move = string(round(median_displacement(pts, pts0); sigdigits = 3)),
      continuous_mmd = string(round(SPlit._exact_mmd(gk, pts, X); sigdigits = 3)),
      rows_kept = "$kept/$(length(sel))",
      test_mmd = test_mmd,
      test_ed = test_ed,
    ),
  )
  return nothing
end

rows = NamedTuple[]
spacings = NamedTuple[]

for N in [1_000, 10_000], (name, data) in datasets(N, MersenneTwister(2026))
  process_dataset(name, data, N, rows, spacings)
end

fmt(x::Float64) = string(round(x; sigdigits = 3))
fmt(x::AbstractString) = x

function markdown_table(rows)
  io = IOBuffer()
  println(
    io,
    "| dataset | N | method | iterations | median move | continuous MMD | rows kept | test-vs-train MMD | test-vs-train energy distance |",
  )
  println(io, "|---|---:|---|---:|---:|---:|---:|---:|---:|")
  for r in rows
    println(
      io,
      "| $(r.dataset) | $(r.N) | $(r.method) | $(r.iterations) | $(r.median_move) | $(r.continuous_mmd) | $(r.rows_kept) | $(fmt(r.test_mmd)) | $(fmt(r.test_ed)) |",
    )
  end
  return String(take!(io))
end

function spacing_table(spacings)
  io = IOBuffer()
  println(io, "| dataset | N | median nearest-neighbor spacing |")
  println(io, "|---|---:|---:|")
  for s in spacings
    println(io, "| $(s.dataset) | $(s.N) | $(fmt(s.nn_spacing)) |")
  end
  return String(take!(io))
end

table = markdown_table(rows)
spacing = spacing_table(spacings)

println(table)
println()
println(spacing)

out_path = joinpath(OUT, "rounding.md")
open(out_path, "w") do io
  println(
    io,
    """
    # Rounding-step experiment

    `SupportPointSplitter` optimizes continuous support points and then maps
    each one to its nearest unclaimed data row (`select_nearest`, sequential
    nearest-neighbor, Joseph & Vakayil 2021). Points are initialized at a
    random sample of data rows, jittered by 0.1% of the per-dimension range
    (`_initial_points`). This experiment measures, per (dataset, N), how far
    the optimizer moves the points relative to the spacing between data rows,
    whether that leaves the rounded selection identical to the initial
    sample, and whether starting away from data rows (uniform in the
    bounding box, or heavy jitter around the initial sample) changes the
    outcome. It also reproduces `run.jl`'s own `datasplit` path for
    `support points · gaussian` (rows marked "(datasplit path)"): `datasplit`
    resolves the `:median` bandwidth from the splitter's `rng` before
    `_initial_points` draws from it, so its initial sample is a different
    draw from the fresh-`rng` "initial sample" row above it. `*` marks
    iteration counts that hit `max_iterations` without converging. `rows
    kept` counts selected rows shared with the relevant initial sample (the
    fresh-`rng` one, or `init_ds` for the "(datasplit path)" row) out of the
    method's own subset size.
    """,
  )
  println(io, table)
  println(io)
  println(io, spacing)
end
println("\nWrote ", out_path)

# ---- figure: fraction of the initial sample's rows kept after rounding,
# one panel per N, grouped by dataset, for the two `support points ·
# gaussian`/`support points · energy` methods (the ones the main table in
# `run.jl` reports).
function kept_fraction(rows, dataset, N, method)
  r =
    only(filter(row -> row.dataset == dataset && row.N == N && row.method == method, rows))
  kept, total = parse.(Int, split(r.rows_kept, "/"))
  return kept / total
end

dataset_order = unique(r.dataset for r in rows)
sp_methods = ["support points · energy", "support points · gaussian"]
sp_colors = Makie.wong_colors()[1:2]
ns = unique(r.N for r in rows)

fig4 = Figure(size = (900, 400))
for (j, N) in enumerate(ns)
  ax = Axis(
    fig4[1, j],
    title = "N = $N",
    xticks = (1:length(dataset_order), dataset_order),
    ylabel = j == 1 ? "rows kept from initial sample" : "",
    limits = (nothing, (0, 1)),
  )
  for (k, m) in enumerate(sp_methods)
    fracs = [kept_fraction(rows, d, N, m) for d in dataset_order]
    barplot!(
      ax,
      (1:length(dataset_order)) .+ (k == 1 ? -0.2 : 0.2),
      fracs;
      width = 0.4,
      color = sp_colors[k],
      label = m,
    )
  end
  j == 1 && axislegend(ax; position = :rt)
end
fig_path = joinpath(OUT, "rounding.png")
save(fig_path, fig4; px_per_unit = 2)
println("Wrote ", fig_path)
