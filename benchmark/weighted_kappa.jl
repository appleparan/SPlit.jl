# Weighted-kappa rule experiment (roadmap M1 open question): with sample
# weights, should the stochastic MM draw its `kappa` rows uniformly and
# rescale their weights (`:uniform`), or draw them in proportion to the
# weights and treat the subsample as uniform (`:proportional`)? For each
# rule, dataset, weight profile, kappa, and rng seed, run the energy-kernel
# support points, round them to rows with `select_nearest`, and score the
# selected rows (uniform) against the full data under the weights with the
# weighted energy distance; record wall time too. Writes
# `docs/src/assets/benchmarks/weighted_kappa.md` and prints the decision.
# Run: `julia -t auto --project=benchmark benchmark/weighted_kappa.jl`.

using SPlit, DataFrames, Random, Statistics

include(joinpath(@__DIR__, "datasets.jl"))

const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

const N = 10_000
const RATIO = 0.2
const KAPPAS = (500, 2_000)
const SEEDS = 1:5
const RULES = (:uniform, :proportional)

# Weight profiles: log-normal (heavy-tailed quality scores) and a 10:1
# two-cluster profile keyed on the sign of the first coordinate.
profiles(data, rng) = [
  ("lognormal", exp.(randn(rng, size(data, 1)))),
  ("cluster-10:1", [x > 0 ? 10.0 : 1.0 for x in view(data, :, 1)]),
]

rows = DataFrame(
  dataset = String[],
  profile = String[],
  kappa = Int[],
  rule = String[],
  mean_discrepancy = Float64[],
  se_discrepancy = Float64[],
  mean_seconds = Float64[],
)

for (dname, data) in datasets(N, MersenneTwister(2026))
  dname in ("normal-10d", "uniform-5d") || continue
  for (pname, w) in profiles(data, MersenneTwister(99))
    X = SPlit.preprocess(data, w)
    n_small = round(Int, RATIO * N)
    for kappa in KAPPAS, rule in RULES
      ds = Float64[]
      ts = Float64[]
      for seed in SEEDS
        t = @elapsed begin
          pts, _, _ = SPlit.support_points(
            EnergyKernel(),
            X,
            n_small;
            kappa,
            weights = w,
            rng = MersenneTwister(seed),
            _subsampling = rule,
          )
          small = SPlit.select_nearest(X, pts)
        end
        push!(ds, energydistance(X[small, :], X; weights_y = w))
        push!(ts, t)
      end
      push!(
        rows,
        (dname, pname, kappa, string(rule), mean(ds), std(ds) / sqrt(length(ds)), mean(ts)),
      )
    end
  end
end

open(joinpath(OUT, "weighted_kappa.md"), "w") do io
  println(
    io,
    "| dataset | profile | kappa | rule | weighted ED (mean ± se, 5 seeds) | mean seconds |",
  )
  println(io, "|---|---|---:|---|---:|---:|")
  for r in eachrow(rows)
    println(
      io,
      "| $(r.dataset) | $(r.profile) | $(r.kappa) | `$(r.rule)` | ",
      "$(round(r.mean_discrepancy; sigdigits = 3)) ± $(round(r.se_discrepancy; sigdigits = 2)) | ",
      "$(round(r.mean_seconds; sigdigits = 3)) |",
    )
  end
end

# Decision: mean discrepancy over all datasets and profiles at kappa = 500.
at500 = rows[rows.kappa.==500, :]
score(rule) = mean(at500[at500.rule.==rule, :mean_discrepancy])
se(rule) =
  sqrt(sum(abs2, at500[at500.rule.==rule, :se_discrepancy])) / count(==(rule), at500.rule)
for rule in RULES
  println(
    "$(rule): mean weighted ED at kappa = 500 = $(score(string(rule))) (se $(se(string(rule))))",
  )
end
gap = score("uniform") - score("proportional")
if gap > se("uniform") + se("proportional")
  println("decision: :proportional (lower by $(gap), beyond one standard error)")
else
  println(
    "decision: :uniform (difference $(gap) within one standard error, simpler rule wins)",
  )
end
