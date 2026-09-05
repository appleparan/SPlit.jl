# Wall time of the twinning and select_nearest search structures — the k-d
# tree and BruteTree from NearestNeighbors, and the plain-matrix
# `SPlit.MatrixSearch` (src/kdtree_selection.jl) — on standard-normal data
# through `SPlit.preprocess`. Decides whether `:brute_tree` is worth keeping
# next to `:matrix` (Design experiments page, "Matrix brute force") and sets
# `SPlit.NEAREST_BRUTE_FORCE_DIMENSION`. Same conventions as
# `benchmark/twinning_trees.jl`: serial, minimum wall time over repeats,
# `--quick`.
#
# Tables 1 and 2 warm each structure up once before timing and discard that
# call, as usual. Table 1 additionally reports that warm-up call's elapsed
# time as "first call", separately per width `p` — a k-d tree/BruteTree
# recompiles its search code for every distinct `p` (it specializes on
# `SVector{p, Float64}`), while `MatrixSearch` does not (it is a plain
# `Matrix{Float64}` regardless of `p`), so "first call" is where the
# 22 s-to-compiler-failure blowup at p >= 1,536 (issue #72) actually shows
# up. Each width is warmed up only once, in this same process, and every
# narrower/wider timing below it reuses that compiled code. Table 2 reports
# the same first-call time per row (500-row/100-point slice at that width)
# for `select_nearest`, since a single `datasplit` call pays that
# compilation cost once. Table 3 repeats just the first-call measurement at
# the widths that used to fail (`:matrix` only — BruteTree/KDTree cannot
# compile there in practical time).
#
# `--table N` (N = 1, 2, or 3) runs only that table, writing
# `docs/src/assets/benchmarks/brute_force_table<N>.md` instead of the
# combined `brute_force.md`. Without `--table`, all three tables run and are
# written together to `docs/src/assets/benchmarks/brute_force.md` (or
# `brute_force_quick.md` under `--quick`). Run:
# `julia --project=benchmark benchmark/brute_force.jl [--quick] [--table N]`.

using SPlit, Random, Statistics

const QUICK = "--quick" in ARGS
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

const TABLE = let i = findfirst(==("--table"), ARGS)
  i === nothing ? nothing : parse(Int, ARGS[i+1])
end
TABLE === nothing ||
  TABLE in (1, 2, 3) ||
  throw(ArgumentError("--table must be 1, 2, or 3, got $TABLE"))

const OUTFILE = if TABLE === nothing
  QUICK ? "brute_force_quick.md" : "brute_force.md"
else
  "brute_force_table$(TABLE).md"
end

repeats(N) = QUICK ? 1 : (N <= 10_000 ? 3 : 1)

io = IOBuffer()
function emit(line)
  println(line)
  flush(stdout)
  println(io, line)
end

noisy_points(rng, data, n) = data[1:n, :] .+ 0.1 .* randn(rng, n, size(data, 2))

# ---------------------------------------------------------------------------
# Table 1: twinning search structures.
# ---------------------------------------------------------------------------

function twin_time(X, N, search)
  return minimum(
    (@elapsed SPlit._twin_groups(X, N ÷ 5, 1, MersenneTwister(0); search)) for
    _ = 1:repeats(N)
  )
end

function run_table1()
  T1_SIZES = QUICK ? [500, 1_000] : [1_000, 10_000]
  T1_DIMS = QUICK ? [50, 200] : [50, 200, 768]
  T1_SEARCHES = (:kdtree, :brute_tree, :matrix)

  emit("## Twinning: search structure wall time")
  emit("")
  emit("First call: one warm-up per `p`, on a 500-row/100-group slice, in this process.")
  emit("")
  emit(
    "| p | k-d tree first call (s) | brute tree first call (s) | matrix first call (s) |",
  )
  emit("|---:|---:|---:|---:|")
  for p in T1_DIMS
    Xwarm = SPlit.preprocess(randn(MersenneTwister(1000 * p), 500, p))
    firsts = [
      (@elapsed SPlit._twin_groups(Xwarm, 100, 1, MersenneTwister(0); search)) for
      search in T1_SEARCHES
    ]
    emit("| $p | $(join(round.(firsts; sigdigits = 3), " | ")) |")
  end

  emit("")
  emit(
    "| N | p | k-d tree (s) | brute tree (s) | matrix (s) | brute/matrix | kdtree/matrix |",
  )
  emit("|---:|---:|---:|---:|---:|---:|---:|")
  for p in T1_DIMS, N in T1_SIZES
    X = SPlit.preprocess(randn(MersenneTwister(1000 * p + round(Int, log10(N))), N, p))
    t_kd, t_bt, t_mx = (twin_time(X, N, search) for search in T1_SEARCHES)
    line = "| $N | $p | $(round(t_kd; sigdigits = 3)) | $(round(t_bt; sigdigits = 3)) | $(round(t_mx; sigdigits = 3)) | $(round(t_bt / t_mx; sigdigits = 3)) | $(round(t_kd / t_mx; sigdigits = 3)) |"
    emit(line)
  end
  if !QUICK
    # Single run at the largest N (minutes per structure at p = 768; keep to p = 50).
    p, N = 50, 100_000
    X = SPlit.preprocess(randn(MersenneTwister(1000 * p + 5), N, p))
    t_kd = @elapsed SPlit._twin_groups(X, N ÷ 5, 1, MersenneTwister(0); search = :kdtree)
    t_bt =
      @elapsed SPlit._twin_groups(X, N ÷ 5, 1, MersenneTwister(0); search = :brute_tree)
    t_mx = @elapsed SPlit._twin_groups(X, N ÷ 5, 1, MersenneTwister(0); search = :matrix)
    line = "| $N | $p | $(round(t_kd; sigdigits = 3)) | $(round(t_bt; sigdigits = 3)) | $(round(t_mx; sigdigits = 3)) | $(round(t_bt / t_mx; sigdigits = 3)) | $(round(t_kd / t_mx; sigdigits = 3)) |"
    emit(line)
  end
  return nothing
end

# ---------------------------------------------------------------------------
# Table 2: select_nearest search structures. `points` are `n` data rows plus
# N(0, 0.1) noise (std 0.1) — close to their own row, like an optimizer's
# output near convergence.
# ---------------------------------------------------------------------------

function select_time(data, points, search)
  m = min(200, size(data, 1))
  SPlit.select_nearest(data[1:m, :], points[1:min(20, size(points, 1)), :]; search)  # warm-up
  return minimum(
    (@elapsed SPlit.select_nearest(data, points; search)) for _ = 1:repeats(size(data, 1))
  )
end

# First-call time for `search` at width `p`: a fresh 500-row slice with 100
# points, so the k-d tree's per-p specialization cost (or MatrixSearch's
# one-time compile) is measured on its own, separately from the repeated
# timed calls below.
function firstcall_time(p, search)
  rng = MersenneTwister(9_000 * p + (search === :kdtree ? 11 : 13))
  data = SPlit.preprocess(randn(rng, 500, p))
  points = noisy_points(rng, data, 100)
  return @elapsed SPlit.select_nearest(data, points; search)
end

function run_table2()
  T2_N = QUICK ? 2_000 : 10_000
  T2_DIMS = QUICK ? [2, 10] : [2, 10, 50, 200, 768]
  T2_N_POINTS = round(Int, 0.2 * T2_N)   # matches the plan's N = 10,000, n = 2,000 ratio
  T2_EXTRA = QUICK ? Tuple{Int,Int}[] : [(100_000, 10), (100_000, 50)]
  T2_SEARCHES = (:kdtree, :matrix)

  emit("")
  emit("## select_nearest: search structure wall time")
  emit("")
  emit("First call: one warm-up per row, on a 500-row/100-point slice, in this process.")
  emit("")
  emit(
    "| N | p | k-d tree first call (s) | matrix first call (s) | k-d tree (s) | matrix (s) | kdtree/matrix |",
  )
  emit("|---:|---:|---:|---:|---:|---:|---:|")
  for p in T2_DIMS
    rng = MersenneTwister(2000 * p)
    data = SPlit.preprocess(randn(rng, T2_N, p))
    points = noisy_points(rng, data, T2_N_POINTS)
    fc_kd, fc_mx = (firstcall_time(p, search) for search in T2_SEARCHES)
    t_kd, t_mx = (select_time(data, points, search) for search in T2_SEARCHES)
    line = "| $T2_N | $p | $(round(fc_kd; sigdigits = 3)) | $(round(fc_mx; sigdigits = 3)) | $(round(t_kd; sigdigits = 3)) | $(round(t_mx; sigdigits = 3)) | $(round(t_kd / t_mx; sigdigits = 3)) |"
    emit(line)
  end
  for (N, p) in T2_EXTRA
    rng = MersenneTwister(2000 * p + 7)
    data = SPlit.preprocess(randn(rng, N, p))
    points = noisy_points(rng, data, round(Int, 0.2 * N))
    fc_kd, fc_mx = (firstcall_time(p, search) for search in T2_SEARCHES)
    t_kd = @elapsed SPlit.select_nearest(data, points; search = :kdtree)   # single run
    t_mx = @elapsed SPlit.select_nearest(data, points; search = :matrix)
    line = "| $N | $p | $(round(fc_kd; sigdigits = 3)) | $(round(fc_mx; sigdigits = 3)) | $(round(t_kd; sigdigits = 3)) | $(round(t_mx; sigdigits = 3)) | $(round(t_kd / t_mx; sigdigits = 3)) |"
    emit(line)
  end
  return nothing
end

# ---------------------------------------------------------------------------
# Table 3: first-call wall time at the widths that used to fail to compile
# (BruteTree/KDTree, issue #72). :matrix only.
# ---------------------------------------------------------------------------

function run_table3()
  T3_DIMS = QUICK ? [3_072] : [3_072, 6_144, 12_288]
  T3_N = 200
  T3_N_SELECTED = 20

  emit("")
  emit(
    "## First call at extreme width (`:matrix` only — the widths BruteTree/KDTree could not compile)",
  )
  emit("")
  emit("| p | twinning first call (s) | select_nearest first call (s) |")
  emit("|---:|---:|---:|")
  for p in T3_DIMS
    rng = MersenneTwister(3000 * p)
    X = SPlit.preprocess(randn(rng, T3_N, p))
    t_twin =
      @elapsed SPlit._twin_groups(X, T3_N_SELECTED, 1, MersenneTwister(0); search = :matrix)
    points = noisy_points(rng, X, T3_N_SELECTED)
    t_sel = @elapsed SPlit.select_nearest(X, points; search = :matrix)
    emit("| $p | $(round(t_twin; sigdigits = 3)) | $(round(t_sel; sigdigits = 3)) |")
  end
  return nothing
end

(TABLE === nothing || TABLE == 1) && run_table1()
(TABLE === nothing || TABLE == 2) && run_table2()
(TABLE === nothing || TABLE == 3) && run_table3()

write(joinpath(OUT, OUTFILE), String(take!(io)))
println("wrote $(joinpath(OUT, OUTFILE))")
