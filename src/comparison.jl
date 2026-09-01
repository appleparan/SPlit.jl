"""
Side-by-side comparison of splitter configurations on one dataset.
"""

using DataFrames

"""
    SplitComparison

Result of [`compare`](@ref): the splitters, their results, and their
[`splitquality`](@ref) values, index-aligned.
"""
struct SplitComparison
  methods::Vector{SupportPointSplitter}
  results::Vector{SplitResult}
  qualities::Vector{Float64}
end

"""
    compare(methods, data; kwargs...) -> SplitComparison

Run [`datasplit`](@ref) with each splitter in `methods` on `data` and score
every split with [`splitquality`](@ref). Keyword arguments are forwarded to
`splitquality`.
"""
function compare(methods::Vector{<:SupportPointSplitter}, data; kwargs...)
  results = [datasplit(m, data) for m in methods]
  qualities = [splitquality(data, r; kwargs...) for r in results]
  return SplitComparison(collect(methods), results, qualities)
end

"""
    DataFrame(comparison::SplitComparison) -> DataFrame

One row per splitter: kernel, ratio, subset sizes, convergence report, and
energy distance (lower is better).
"""
function DataFrames.DataFrame(c::SplitComparison)
  return DataFrame(
    kernel = [string(nameof(typeof(m.kernel))) for m in c.methods],
    ratio = [m.ratio for m in c.methods],
    train = [length(r.train_indices) for r in c.results],
    test = [length(r.test_indices) for r in c.results],
    converged = [r.converged for r in c.results],
    iterations = [r.iterations for r in c.results],
    energy_distance = c.qualities,
  )
end

"""
    best(comparison::SplitComparison) -> (method, result)

The splitter/result pair with the lowest energy distance.
"""
function best(c::SplitComparison)
  i = argmin(c.qualities)
  return c.methods[i], c.results[i]
end

function Base.show(io::IO, c::SplitComparison)
  println(io, "SplitComparison with $(length(c.methods)) methods:")
  show(io, DataFrame(c); allrows = true, allcols = true)
end
