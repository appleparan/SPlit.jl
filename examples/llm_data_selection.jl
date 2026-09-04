# Selecting LLM training data from an embedding matrix with SPlit.jl.
#
# Downloads a public arXiv-abstract embedding dataset (5,000 abstracts,
# MiniLM 384-d; CC0), cosine-normalizes the rows, and selects n = 500
# abstracts (10%) with every splitter under three target measures — the
# data itself, a quality-weighted version (abstract length as a stand-in),
# and a target sub-population (the `cs` archive) — against uniform random
# and K-center greedy baselines. Also times Compress++ against plain kernel
# thinning for n = 250 ≪ N. Prints a markdown table and writes it to
# docs/src/assets/examples/llm_selection.md.
#
# Setup (from the repository root; path="." is the checkout):
#   julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia -t auto --project=examples examples/llm_data_selection.jl
#
# Run (a few minutes): julia -t auto --project=examples examples/llm_data_selection.jl
# Options: --model minilm|arcticlarge, --out PATH, --n 500

using SPlit, DataFrames, DuckDB, Downloads, LinearAlgebra, Printf, Random, Statistics

# Value of `flag`, or `default` when the flag is absent; a trailing flag
# without a value is an error rather than a `BoundsError`.
function argvalue(flag, default)
  i = findfirst(==(flag), ARGS)
  i === nothing && return default
  i < length(ARGS) || error("$flag needs a value")
  return ARGS[i+1]
end

const MODEL = argvalue("--model", "minilm")
const N_SELECT = parse(Int, argvalue("--n", "500"))
const OUT = argvalue(
  "--out",
  joinpath(@__DIR__, "..", "docs", "src", "assets", "examples", "llm_selection.md"),
)
const DATASET = "https://huggingface.co/datasets/sondalex/arxiv-abstracts-2021-embeddings-10000/resolve/main/data/arxiv-abstract-$(MODEL).parquet"

# ---- data
datadir = joinpath(@__DIR__, "data")
mkpath(datadir)
file = joinpath(datadir, "arxiv-abstract-$(MODEL).parquet")
isfile(file) || (println("downloading $DATASET"); Downloads.download(DATASET, file))
con = DBInterface.connect(DuckDB.DB)
df = DataFrame(
  DBInterface.execute(
    con,
    "SELECT categories, length(content) AS len, embedding FROM read_parquet('$file')",
  ),
)
E = Matrix{Float64}(reduce(hcat, [Float64.(coalesce.(e, 0.0)) for e in df.embedding])')
E ./= norm.(eachrow(E))                                   # cosine-normalize
N, p = size(E)
w = min.(Float64.(df.len), quantile(Float64.(df.len), 0.99))   # quality proxy: abstract length, clipped
is_cs = [any(==("cs"), string.(c)) for c in df.categories]
R = E[is_cs, :]
println("N = $N, p = $p, target rows (cs) = $(size(R, 1)), n = $N_SELECT")

# ---- baselines
random_rows(n, rng) = randperm(rng, N)[1:n]
function kcenter_greedy(E, n, rng)                       # farthest-first traversal (Sener & Savarese 2018)
  N = size(E, 1)
  sel = [rand(rng, 1:N)]
  mind = fill(Inf, N)
  for _ = 2:n
    last = sel[end]
    @views for i = 1:N
      mind[i] = min(mind[i], norm(E[i, :] .- E[last, :]))
    end
    push!(sel, argmax(mind))
  end
  return sel
end

# ---- scoring: energy distance of the selection to the measure the setting optimizes, plus to the plain data
score_plain(sel) = energydistance(E[sel, :], E)
score_weighted(sel) = energydistance(E[sel, :], E; weights_y = w)
score_target(sel) = energydistance(E[sel, :], R)

rows = DataFrame(
  setting = String[],
  method = String[],
  optimized = Float64[],
  plain = Float64[],
  seconds = Float64[],
)
function record!(setting, method, sel, seconds, scorer)
  push!(rows, (setting, method, scorer(sel), score_plain(sel), seconds))
end

splitters(seed) = [
  (
    "herding · energy",
    HerdingSplitter(kernel = EnergyKernel(), rng = MersenneTwister(seed)),
  ),
  ("twinning", TwinningSplitter()),
  ("kernel thinning · energy", KernelThinningSplitter(rng = MersenneTwister(seed))),
  (
    "support points · energy",
    SupportPointSplitter(kappa = 1_000, max_iterations = 100, rng = MersenneTwister(seed)),
  ),
]

for (setting, kwargs, scorer, skip) in (
  ("plain", (;), score_plain, String[]),
  ("weights = length", (; weights = w), score_weighted, ["twinning"]),
  ("reference = cs", (; reference = R), score_target, ["twinning"]),
)
  # random: mean of 5 seeds
  rs = [random_rows(N_SELECT, MersenneTwister(100 + i)) for i = 1:5]
  push!(rows, (setting, "random", mean(scorer.(rs)), mean(score_plain.(rs)), 0.0))
  kcenter_greedy(E[1:200, :], 20, MersenneTwister(0))   # warm-up (JIT)
  t = @elapsed sel = kcenter_greedy(E, N_SELECT, MersenneTwister(7))
  record!(setting, "k-center greedy", sel, t, scorer)
  # separate rng seeds so the warm-up run (compilation only, on a throwaway
  # splitter copy) never consumes the timed splitter's own rng stream
  for ((label, s_warmup), (_, s)) in zip(splitters(0), splitters(1))
    label in skip && continue
    # weights must match the sliced row count; reference is unaffected by the slice
    warmup_kwargs =
      haskey(kwargs, :weights) ? merge(kwargs, (; weights = kwargs.weights[1:200])) : kwargs
    selectrows(s_warmup, E[1:200, :], 20; standardize = false, warmup_kwargs...)   # warm-up (JIT)
    t = @elapsed sel = selectrows(s, E, N_SELECT; standardize = false, kwargs...)
    record!(setting, label, sel, t, scorer)
  end
end

# ---- Compress++ against plain kernel thinning at n ≪ N
let n = 250
  for (label, mode) in (
    ("kernel thinning · compress = :never", :never),
    ("kernel thinning · compress = :always", :always),
  )
    # a throwaway splitter for the warm-up, so the timed one starts on a fresh rng
    selectrows(
      KernelThinningSplitter(compress = mode, rng = MersenneTwister(0)),
      E[1:400, :],
      20;
      standardize = false,
    )
    s = KernelThinningSplitter(compress = mode, rng = MersenneTwister(3))
    t = @elapsed sel = selectrows(s, E, n; standardize = false)
    record!("plain, n = $n", label, sel, t, score_plain)
  end
  rs = [random_rows(n, MersenneTwister(200 + i)) for i = 1:5]
  push!(
    rows,
    ("plain, n = $n", "random", mean(score_plain.(rs)), mean(score_plain.(rs)), 0.0),
  )
end

# ---- table
io = IOBuffer()
println(
  io,
  "| setting | method | energy distance to the optimized measure | energy distance to the data | seconds |",
)
println(io, "|---|---|---:|---:|---:|")
for r in eachrow(rows)
  @printf(
    io,
    "| %s | %s | %.3g | %.3g | %s |\n",
    r.setting,
    r.method,
    r.optimized,
    r.plain,
    r.method == "random" ? "–" : @sprintf("%.2g", r.seconds)
  )
end
table = String(take!(io))
print(table)
mkpath(dirname(OUT))
write(OUT, table)
println("wrote $OUT")
