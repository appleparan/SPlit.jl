"""
Comparison and benchmarking utilities for different splitting methods.
"""

using Statistics
using DataFrames

"""
    SplitComparison

Container for comparing multiple splitting methods.
"""
struct SplitComparison
  methods::Vector{SplittingMethod}
  results::Vector{SplitResult}
  data_info::NamedTuple
end

"""
    compare(methods::Vector{<:SplittingMethod}, data; quality::Bool=true) -> SplitComparison

Compare multiple splitting methods on the same dataset.

# Arguments
- `methods`: Vector of splitting methods to compare
- `data`: Dataset to split
- `quality`: Whether to compute quality metrics

# Returns
- `SplitComparison` object with results and analysis

# Examples
```julia
methods = [
    SupportPointSplitter(Euclidean(); ratio=0.2),
    SupportPointSplitter(Cityblock(); ratio=0.2),
    SupportPointSplitter(EnergyDistance(Euclidean()); ratio=0.2)
]

comparison = compare(methods, data)
summary(comparison)
```
"""
function compare(methods::Vector{<:SplittingMethod}, data; quality::Bool = true)
  results = if quality
    [split_with_quality(method, data) for method in methods]
  else
    [split(method, data) for method in methods]
  end

  data_info = (size = size(data), type = typeof(data), n_total = size(data, 1))

  return SplitComparison(methods, results, data_info)
end

"""
    summary(comparison::SplitComparison) -> DataFrame

Generate a summary table of splitting method comparison.
"""
function Base.summary(comparison::SplitComparison)
  df = DataFrame(
    Method = String[],
    Metric = String[],
    Ratio = Float64[],
    TrainSize = Int[],
    TestSize = Int[],
    Quality = Union{Float64,Missing}[],
    Converged = Bool[],
    Iterations = Int[],
  )

  for (method, result) in zip(comparison.methods, comparison.results)
    push!(
      df,
      (
        Method = string(typeof(method)),
        Metric = string(typeof(metric(method))),
        Ratio = ratio(method),
        TrainSize = length(result.train_indices),
        TestSize = length(result.test_indices),
        Quality = result.quality,
        Converged = result.convergence,
        Iterations = result.iterations,
      ),
    )
  end

  return df
end

"""
    best(comparison::SplitComparison; by::Symbol=:Quality) -> Tuple{SplittingMethod, SplitResult}

Find the best splitting method from comparison.

# Arguments
- `comparison`: SplitComparison object
- `by`: Criterion for selection (:Quality, :TrainSize, :TestSize)
"""
function best(comparison::SplitComparison; by::Symbol = :Quality)
  if by === :Quality
    qualities = [r.quality for r in comparison.results]
    if any(q -> q !== nothing, qualities)
      # Lower energy distance is better
      idx = argmin(filter(!isnothing, qualities))
      non_nothing_indices = findall(q -> q !== nothing, qualities)
      best_idx = non_nothing_indices[idx]
    else
      throw(ArgumentError("No quality metrics available for comparison"))
    end
  elseif by === :TrainSize
    idx = argmax([length(r.train_indices) for r in comparison.results])
    best_idx = idx
  elseif by === :TestSize
    idx = argmax([length(r.test_indices) for r in comparison.results])
    best_idx = idx
  else
    throw(ArgumentError("Unknown criterion: $by"))
  end

  return comparison.methods[best_idx], comparison.results[best_idx]
end

# Pretty printing
function Base.show(io::IO, comparison::SplitComparison)
  println(io, "SplitComparison with $(length(comparison.methods)) methods:")
  println(io, "  Data: $(comparison.data_info.type) $(comparison.data_info.size)")
  println(io, "  Total samples: $(comparison.data_info.n_total)")

  df = summary(comparison)
  show(io, df, allrows = true, allcols = true)
end

"""
    DefaultSplitters(ratio::Float64=0.2) -> Vector{SupportPointSplitter}

Create a set of default splitting methods for comparison.
"""
function DefaultSplitters(ratio::Float64 = 0.2)
  return [
    SupportPointSplitter(Euclidean(); ratio = ratio),
    SupportPointSplitter(Cityblock(); ratio = ratio),
    SupportPointSplitter(EnergyDistance(Euclidean()); ratio = ratio),
  ]
end

"""
    quick_compare(data; ratio::Float64=0.2, max_iterations::Int=50) -> SplitComparison

Quick comparison using default methods with reduced iterations for speed.
"""
function quick_compare(data; ratio::Float64 = 0.2, max_iterations::Int = 50)
  methods = [
    SupportPointSplitter(Euclidean(); ratio = ratio, max_iterations = max_iterations),
    SupportPointSplitter(Cityblock(); ratio = ratio, max_iterations = max_iterations),
    SupportPointSplitter(
      EnergyDistance(Euclidean());
      ratio = ratio,
      max_iterations = max_iterations,
    ),
  ]

  return compare(methods, data; quality = true)
end
