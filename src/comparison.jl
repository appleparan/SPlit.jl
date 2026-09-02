"""
Side-by-side comparison of splitter configurations on one dataset.
"""

using DataFrames
using Random

"""
    SplitComparison

Result of [`compare`](@ref): the fitted splitters (with any `:median`
bandwidth resolved from the data), their results, their
[`splitquality`](@ref) values, and the `kernel` they were scored under,
index-aligned.
"""
struct SplitComparison
  methods::Vector{SupportPointSplitter}
  results::Vector{SplitResult}
  qualities::Vector{Float64}
  kernel::SplitKernel
end
SplitComparison(m, r, q) = SplitComparison(m, r, q, EnergyKernel())

"""
    compare(methods, data; kernel = EnergyKernel(), kwargs...) -> SplitComparison

Run [`datasplit`](@ref) with each splitter in `methods` on `data` and score
every split with [`splitquality`](@ref) under `kernel`. `kernel` selects the
scoring discrepancy; a `:median` bandwidth is resolved once on the
preprocessed data with `rng` and the resolved kernel is stored in the
comparison. Remaining keyword arguments are forwarded to `splitquality`.
"""
function compare(
  methods::Vector{<:SupportPointSplitter},
  data;
  kernel::SplitKernel = EnergyKernel(),
  rng::AbstractRNG = Random.default_rng(),
  kwargs...,
)
  results = [datasplit(m, data) for m in methods]
  k = isresolved(kernel) ? kernel : resolve(kernel, preprocess(data), rng)
  qualities = [splitquality(data, r; kernel = k, rng, kwargs...) for r in results]
  return SplitComparison([r.method for r in results], results, qualities, k)
end

"""
    DataFrame(comparison::SplitComparison) -> DataFrame

One row per splitter: kernel, ratio, subset sizes, convergence report, and
the discrepancy score (`energy_distance` or `mmd`, lower is better).
"""
function DataFrames.DataFrame(c::SplitComparison)
  score = c.kernel isa EnergyKernel ? :energy_distance : :mmd
  return DataFrame(
    :kernel => [string(nameof(typeof(m.kernel))) for m in c.methods],
    :ratio => [m.ratio for m in c.methods],
    :train => [length(r.train_indices) for r in c.results],
    :test => [length(r.test_indices) for r in c.results],
    :converged => [r.converged for r in c.results],
    :iterations => [r.iterations for r in c.results],
    score => c.qualities,
  )
end

"""
    best(comparison::SplitComparison) -> (method, result)

The splitter/result pair with the lowest discrepancy.
"""
function best(c::SplitComparison)
  i = argmin(c.qualities)
  return c.methods[i], c.results[i]
end

function Base.show(io::IO, c::SplitComparison)
  println(io, "SplitComparison with $(length(c.methods)) methods:")
  show(io, DataFrame(c); allrows = true, allcols = true)
end
