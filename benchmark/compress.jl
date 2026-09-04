# Wall time and quality cost of Compress++ (`KernelThinningSplitter(compress
# = :always)` vs `:never`) on standard-normal embeddings, for N = 10,000 to
# 100,000, p = 10 to 384, and ratio n/N = 0.01 to 0.20. Decides/validates
# `SPlit._compress_pays_off` (Design experiments page): the rule should fire
# only where compress is actually faster, without giving up much quality.
# Timing is the minimum wall time over three seeds (one in quick mode);
# energy distance is averaged over the same three runs. The estimator
# mirrors `splitquality`'s `exact_threshold` rule (total row count, here
# N + n): `Exact()` at or below 20,000, `SPlit.ENERGY_FALLBACK`
# (`RandomSlices(64)`) above it. Writes
# `docs/src/assets/benchmarks/compress.md`. Run:
# `julia -t auto --project=benchmark benchmark/compress.jl [--quick]` — IS
# threaded (kernel thinning sums its terms in fixed 1,024-row chunks, so
# results are independent of the thread count).

using SPlit, Random, Statistics

const QUICK = "--quick" in ARGS
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

const CELLS =
  QUICK ? [(2_000, 10), (2_000, 32)] : [(10_000, 10), (10_000, 384), (100_000, 10)]
const RATIOS = [0.01, 0.05, 0.10, 0.20]
const OUTFILE = QUICK ? "compress_quick.md" : "compress.md"
const SEEDS = QUICK ? [0] : [0, 1, 2]

splitter(compress, seed) =
  KernelThinningSplitter(; kernel = EnergyKernel(), compress, rng = MersenneTwister(seed))

function timed_quality(X, n, compress, quality)
  selectrows(splitter(compress, 0), X[1:min(size(X, 1), 500), :], 50; standardize = false)  # warm-up
  times = Float64[]
  eds = Float64[]
  for seed in SEEDS
    rows = Int[]
    push!(
      times,
      @elapsed (rows = selectrows(splitter(compress, seed), X, n; standardize = false))
    )
    push!(eds, quality(rows))
  end
  return minimum(times), mean(eds)
end

io = IOBuffer()
println(
  io,
  "| N | p | n | n/N | auto fires | g | plain (s) | compress++ (s) | plain / compress++ | ED plain | ED compress++ | ED random |",
)
println(io, "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
for (N, p) in CELLS
  X = randn(MersenneTwister(1000 * p + round(Int, log10(N))), N, p)
  for r in RATIOS
    n = round(Int, r * N)
    est = N + n <= 20_000 ? Exact() : SPlit.ENERGY_FALLBACK
    quality(rows) = energydistance(X[rows, :], X; estimator = est, rng = MersenneTwister(1))
    t_plain, ed_plain = timed_quality(X, n, :never, quality)
    t_compress, ed_compress = timed_quality(X, n, :always, quality)
    fires = SPlit._compress_pays_off(N, n)
    g = SPlit._compress_g(N, n)
    ed_random =
      mean(quality(randperm(MersenneTwister(100 + seed), N)[1:n]) for seed in SEEDS)
    line = "| $N | $p | $n | $(round(r; sigdigits = 3)) | $(fires ? "yes" : "no") | $g | $(round(t_plain; sigdigits = 3)) | $(round(t_compress; sigdigits = 3)) | $(round(t_plain / t_compress; sigdigits = 3)) | $(round(ed_plain; sigdigits = 3)) | $(round(ed_compress; sigdigits = 3)) | $(round(ed_random; sigdigits = 3)) |"
    println(line)
    flush(stdout)
    println(io, line)
  end
end
write(joinpath(OUT, OUTFILE), String(take!(io)))
println("wrote $(joinpath(OUT, OUTFILE))")
