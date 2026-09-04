# Wall time and quality cost of Compress++ (`KernelThinningSplitter(compress
# = :always)` vs `:never`) on standard-normal embeddings, for N = 10,000 to
# 100,000, p = 10 to 384, and ratio n/N = 0.01 to 0.20. Decides/validates
# `SPlit._compress_pays_off` (Design experiments page): the rule should fire
# only where compress is actually faster, without giving up much quality.
# Writes `docs/src/assets/benchmarks/compress.md`. Run:
# `julia -t auto --project=benchmark benchmark/compress.jl [--quick]` — IS
# threaded (kernel thinning sums its terms in fixed 1,024-row chunks, so
# results are independent of the thread count).

using SPlit, Random

const QUICK = "--quick" in ARGS
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

const CELLS =
  QUICK ? [(2_000, 10), (2_000, 32)] : [(10_000, 10), (10_000, 384), (100_000, 10)]
const RATIOS = [0.01, 0.05, 0.10, 0.20]
const OUTFILE = QUICK ? "compress_quick.md" : "compress.md"

repeats(N) = QUICK ? 1 : (N <= 10_000 ? 3 : 1)

splitter(compress) =
  KernelThinningSplitter(; kernel = EnergyKernel(), compress, rng = MersenneTwister(0))

function timed(X, n, compress)
  selectrows(splitter(compress), X[1:min(size(X, 1), 500), :], 50; standardize = false)  # warm-up
  rows = Int[]
  t = minimum(1:repeats(size(X, 1))) do _
    @elapsed (rows = selectrows(splitter(compress), X, n; standardize = false))
  end
  return t, rows
end

io = IOBuffer()
println(
  io,
  "| N | p | n | n/N | auto fires | g | plain (s) | compress++ (s) | plain / compress++ | ED plain | ED compress++ | ED random |",
)
println(io, "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
for (N, p) in CELLS
  X = randn(MersenneTwister(1000 * p + round(Int, log10(N))), N, p)
  est = N <= 20_000 ? Exact() : RandomSlices(64)
  quality(rows) = energydistance(X[rows, :], X; estimator = est, rng = MersenneTwister(1))
  for r in RATIOS
    n = round(Int, r * N)
    t_plain, rows_plain = timed(X, n, :never)
    t_compress, rows_compress = timed(X, n, :always)
    fires = SPlit._compress_pays_off(N, n)
    g = SPlit._compress_g(N, n)
    rand_rows = randperm(MersenneTwister(2), N)[1:n]
    line = "| $N | $p | $n | $(round(r; sigdigits = 3)) | $(fires ? "yes" : "no") | $g | $(round(t_plain; sigdigits = 3)) | $(round(t_compress; sigdigits = 3)) | $(round(t_plain / t_compress; sigdigits = 3)) | $(round(quality(rows_plain); sigdigits = 3)) | $(round(quality(rows_compress); sigdigits = 3)) | $(round(quality(rand_rows); sigdigits = 3)) |"
    println(line)
    flush(stdout)
    println(io, line)
  end
end
write(joinpath(OUT, OUTFILE), String(take!(io)))
println("wrote $(joinpath(OUT, OUTFILE))")
