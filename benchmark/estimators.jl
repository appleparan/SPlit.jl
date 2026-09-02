# Estimator selection experiment: for the split produced by
# `support points · energy`, `herding · energy` (scored by energy distance)
# and `herding · gaussian` (scored by MMD, median bandwidth), measure every
# candidate `DiscrepancyEstimator`'s absolute error against the exact value
# and its wall time, over 5 rng seeds, on the four Phase 2b datasets at
# N = 10,000. Writes `docs/src/assets/benchmarks/estimators.md` and prints
# the decision rule's outcome per kernel. Run:
# `julia -t auto --project=benchmark benchmark/estimators.jl`.

using SPlit, DataFrames, Random, Statistics

include(joinpath(@__DIR__, "datasets.jl"))

const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(OUT)

const N = 10_000
const SEEDS = 1:5
const SUBSAMPLE_LABEL = "Subsample(2000, 8)"

energy_estimators() = [
  (SUBSAMPLE_LABEL, Subsample(2_000, 8)),
  ("RandomSlices(64)", RandomSlices(64)),
  ("RandomSlices(256)", RandomSlices(256)),
  ("RandomSlices(1024)", RandomSlices(1024)),
]

gaussian_estimators() = [
  (SUBSAMPLE_LABEL, Subsample(2_000, 8)),
  ("RandomFeatures(512)", RandomFeatures(512)),
  ("RandomFeatures(2048)", RandomFeatures(2048)),
]

score(::EnergyKernel, train, test, estimator, rng) =
  energydistance(train, test; estimator, rng)
score(kernel::GaussianKernel, train, test, estimator, rng) =
  mmd(train, test, kernel; estimator, rng)

rows = DataFrame(
  dataset = String[],
  split = String[],
  kernel = String[],
  estimator = String[],
  mean_abs_error = Float64[],
  max_abs_error = Float64[],
  mean_seconds = Float64[],
)

for (dname, data) in datasets(N, MersenneTwister(2026))
  X = SPlit.preprocess(data)
  gk = SPlit.resolve(GaussianKernel(), X, MersenneTwister(7))   # once per dataset

  sp_energy = datasplit(
    SupportPointSplitter(kernel = EnergyKernel(), kappa = 1_000, rng = MersenneTwister(11)),
    data,
  )
  herd_energy =
    datasplit(HerdingSplitter(kernel = EnergyKernel(), rng = MersenneTwister(12)), data)
  herd_gauss =
    datasplit(HerdingSplitter(kernel = GaussianKernel(), rng = MersenneTwister(13)), data)

  splits = [
    (
      "support points · energy",
      sp_energy,
      EnergyKernel(),
      "EnergyKernel",
      energy_estimators(),
    ),
    ("herding · energy", herd_energy, EnergyKernel(), "EnergyKernel", energy_estimators()),
    ("herding · gaussian", herd_gauss, gk, "GaussianKernel", gaussian_estimators()),
  ]

  for (split_label, r, kernel, kernel_label, estimators) in splits
    train = X[r.train_indices, :]
    test = X[r.test_indices, :]
    exact_value = score(kernel, train, test, Exact(), MersenneTwister(0))

    for (label, est) in estimators
      score(kernel, train, test, est, MersenneTwister(0))   # warm-up, not timed
      errs = Float64[]
      times = Float64[]
      for s in SEEDS
        rng = MersenneTwister(s)
        t = @elapsed val = score(kernel, train, test, est, rng)
        push!(errs, abs(val - exact_value))
        push!(times, t)
      end
      push!(
        rows,
        (dname, split_label, kernel_label, label, mean(errs), maximum(errs), mean(times)),
      )
    end
  end
end

# ---- detailed table
function markdown_table(df)
  io = IOBuffer()
  println(
    io,
    "| dataset | split | kernel | estimator | mean abs error | max abs error | mean time (s) |",
  )
  println(io, "|---|---|---|---|---:|---:|---:|")
  for r in eachrow(df)
    println(
      io,
      "| $(r.dataset) | $(r.split) | $(r.kernel) | $(r.estimator) | " *
      "$(round(r.mean_abs_error; sigdigits = 3)) | $(round(r.max_abs_error; sigdigits = 3)) | " *
      "$(round(r.mean_seconds; sigdigits = 3)) |",
    )
  end
  return String(take!(io))
end
detail_table = markdown_table(rows)
println(detail_table)

# ---- per-kernel aggregate and decision rule: the cheapest candidate whose
# max error (worst across all dataset/split rows) is at most one third of
# Subsample's, at no more than Subsample's mean wall time; Subsample itself
# otherwise.
function aggregate(rows, kernel_label)
  sub = filter(r -> r.kernel == kernel_label, rows)
  agg =
    DataFrame(estimator = String[], agg_max_error = Float64[], agg_mean_seconds = Float64[])
  for e in unique(sub.estimator)
    esub = filter(r -> r.estimator == e, sub)
    push!(agg, (e, maximum(esub.max_abs_error), mean(esub.mean_seconds)))
  end
  return agg
end

function decide(agg)
  base = only(filter(r -> r.estimator == SUBSAMPLE_LABEL, eachrow(agg)))
  candidates = filter(r -> r.estimator != SUBSAMPLE_LABEL, agg)
  function passes(r)
    within_error = r.agg_max_error <= base.agg_max_error / 3
    within_time = r.agg_mean_seconds <= base.agg_mean_seconds
    return within_error && within_time
  end
  qualifying = filter(passes, candidates)
  isempty(qualifying) && return SUBSAMPLE_LABEL
  return qualifying[argmin(qualifying.agg_mean_seconds), :estimator]
end

function markdown_agg(agg)
  io = IOBuffer()
  println(io, "| estimator | max abs error (worst over rows) | mean time (s) (over rows) |")
  println(io, "|---|---:|---:|")
  for r in eachrow(agg)
    println(
      io,
      "| $(r.estimator) | $(round(r.agg_max_error; sigdigits = 3)) | $(round(r.agg_mean_seconds; sigdigits = 3)) |",
    )
  end
  return String(take!(io))
end

energy_agg = aggregate(rows, "EnergyKernel")
gaussian_agg = aggregate(rows, "GaussianKernel")
energy_decision = decide(energy_agg)
gaussian_decision = decide(gaussian_agg)

println(
  "\nDecision rule: cheapest candidate with max error <= 1/3 Subsample's, ",
  "at <= Subsample's mean time; Subsample otherwise.",
)
println("EnergyKernel decision: $energy_decision")
println("GaussianKernel decision: $gaussian_decision")

out_path = joinpath(OUT, "estimators.md")
open(out_path, "w") do io
  println(
    io,
    """
    # Estimator selection experiment

    Absolute error against the exact value and wall time of every candidate
    [`DiscrepancyEstimator`](@ref) for `energydistance`/`mmd`, measured on the
    splits from `support points · energy`, `herding · energy` (scored by
    energy distance) and `herding · gaussian` (scored by MMD, median
    bandwidth resolved once per dataset), on the four Phase 2b datasets at
    N = 10,000, over 5 rng seeds per (dataset, split, estimator) cell.

    ## Per-row results
    """,
  )
  println(io, detail_table)
  println(
    io,
    """

    ## Decision

    Rule: an estimator becomes the automatic `splitquality` fallback if, at
    no more than `$SUBSAMPLE_LABEL`'s mean wall time, its worst-case max
    error over every (dataset, split) row is at most one third of
    `$SUBSAMPLE_LABEL`'s; otherwise `$SUBSAMPLE_LABEL` stays the fallback.
    Aggregates below take the worst (maximum) max-abs-error and the mean
    mean-time over every row for that kernel.

    ### EnergyKernel
    """,
  )
  println(io, markdown_agg(energy_agg))
  println(io, "\n**Decision: `$energy_decision`**\n")
  println(io, "### GaussianKernel\n")
  println(io, markdown_agg(gaussian_agg))
  println(io, "\n**Decision: `$gaussian_decision`**")
end
println("\nWrote ", out_path)
