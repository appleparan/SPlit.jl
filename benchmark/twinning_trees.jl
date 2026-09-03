# Nearest-neighbor structure for twinning: wall time of `_twin_groups` with
# a k-d tree and with brute force on standard-normal data, ratio 0.2, for
# N = 1,000 to 100,000 and dimensions 2 to 768. Decides
# `SPlit.TWINNING_BRUTE_FORCE_DIMENSION` (Design experiments page) and how
# the crossover moves with N. REPEATS = 3 for N <= 10,000, REPEATS = 1 for
# N = 100,000 (the p = 768 row there takes minutes per structure). Writes
# `docs/src/assets/benchmarks/twinning_trees.md`. Run:
# `julia --project=benchmark benchmark/twinning_trees.jl [--quick]` (serial
# by design; at N = 100,000, p = 768 the data matrix is ~600 MB, fine on a
# 46 GB machine — do not add threading).

using SPlit, Random, Statistics

const QUICK = "--quick" in ARGS
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

const SIZES = QUICK ? [500, 1_000] : [1_000, 10_000, 100_000]
const DIMS = QUICK ? [2, 10, 50] : [2, 10, 50, 200, 768]
const OUTFILE = QUICK ? "twinning_trees_quick.md" : "twinning_trees.md"

repeats(N) = QUICK ? 1 : (N <= 10_000 ? 3 : 1)

function timed(X, N, brute)
  SPlit._twin_groups(X[1:500, :], 100, 1, MersenneTwister(0); brute_force = brute)  # warm-up
  return minimum(
    (@elapsed SPlit._twin_groups(X, N ÷ 5, 1, MersenneTwister(0); brute_force = brute)) for
    _ = 1:repeats(N)
  )
end

io = IOBuffer()
println(io, "| N | p | k-d tree (s) | brute force (s) | brute / k-d |")
println(io, "|---:|---:|---:|---:|---:|")
for N in SIZES, p in DIMS
  X = SPlit.preprocess(randn(MersenneTwister(1000 * p + round(Int, log10(N))), N, p))
  t_kd = timed(X, N, false)
  t_bf = timed(X, N, true)
  line = "| $N | $p | $(round(t_kd; sigdigits = 3)) | $(round(t_bf; sigdigits = 3)) | $(round(t_bf / t_kd; sigdigits = 3)) |"
  println(line)
  flush(stdout)
  println(io, line)
end
write(joinpath(OUT, OUTFILE), String(take!(io)))
println("wrote $(joinpath(OUT, OUTFILE))")
