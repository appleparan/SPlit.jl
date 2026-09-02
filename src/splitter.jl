"""
Public splitting API: `SupportPointSplitter` configuration and `datasplit`.
"""

using DataFrames
using Random

"""
    AbstractSplitter

Supertype of every splitting method usable with [`datasplit`](@ref).
Concrete splitters expose `kernel` and `ratio` fields, which
`compare`/`DataFrame(::SplitComparison)` read.
"""
abstract type AbstractSplitter end

"""
    SupportPointSplitter(; kernel = EnergyKernel(), ratio = 0.2, kappa = nothing,
                          max_iterations = 500, tolerance = 1e-10,
                          n_threads = Threads.nthreads(),
                          rng = Random.default_rng(), verbose = false)

Configuration for optimal data splitting via support points
(Joseph & Vakayil 2021): the smaller side is computed as a set of support
points and mapped to data rows by sequential nearest-neighbor selection.

- `kernel`: `EnergyKernel()` (default) or `GaussianKernel(σ)`; a `:median`
  bandwidth is resolved from the data at `datasplit` time and the resolved
  kernel is stored in `result.method`.
- `ratio`: fraction of rows assigned to the test set, in (0, 1).
- `kappa`: absolute per-iteration subsample size for stochastic optimization;
  `nothing` uses all rows every iteration.
- `tolerance`: convergence when the largest squared displacement of any
  support point in one iteration is below this value. In stochastic mode
  the running-average weight decays with the iteration count, so
  convergence there partly reflects that step-size decay rather than the
  objective flattening out.
- `rng`: source of all randomness (initialization, jitter, stochastic
  subsampling); pass a seeded RNG for reproducible splits.
- `verbose`: print per-iteration progress.
"""
struct SupportPointSplitter{K<:SplitKernel,R<:AbstractRNG} <: AbstractSplitter
  kernel::K
  ratio::Float64
  kappa::Union{Nothing,Int}
  max_iterations::Int
  tolerance::Float64
  n_threads::Int
  rng::R
  verbose::Bool
end

function SupportPointSplitter(;
  kernel::SplitKernel = EnergyKernel(),
  ratio::Real = 0.2,
  kappa::Union{Nothing,Int} = nothing,
  max_iterations::Int = 500,
  tolerance::Real = 1e-10,
  n_threads::Int = Threads.nthreads(),
  rng::AbstractRNG = Random.default_rng(),
  verbose::Bool = false,
)
  ratio = Float64(ratio)
  tolerance = Float64(tolerance)
  0 < ratio < 1 || throw(ArgumentError("ratio must be in (0, 1), got $ratio"))
  kappa === nothing ||
    kappa > 0 ||
    throw(ArgumentError("kappa must be positive, got $kappa"))
  (kernel isa GaussianKernel && kappa !== nothing) &&
    throw(ArgumentError("stochastic mode (kappa) is not available for GaussianKernel yet"))
  max_iterations > 0 ||
    throw(ArgumentError("max_iterations must be positive, got $max_iterations"))
  tolerance > 0 || throw(ArgumentError("tolerance must be positive, got $tolerance"))
  n_threads > 0 || throw(ArgumentError("n_threads must be positive, got $n_threads"))
  return SupportPointSplitter(
    kernel,
    ratio,
    kappa,
    max_iterations,
    tolerance,
    n_threads,
    rng,
    verbose,
  )
end

"""
    SplitResult

Outcome of [`datasplit`](@ref): index partition plus an honest report of the
optimizer's convergence.
"""
struct SplitResult{M<:AbstractSplitter}
  train_indices::Vector{Int}
  test_indices::Vector{Int}
  converged::Bool
  iterations::Int
  method::M
end

"""
    train_indices(result::SplitResult) -> Vector{Int}
"""
train_indices(r::SplitResult) = r.train_indices

"""
    test_indices(result::SplitResult) -> Vector{Int}
"""
test_indices(r::SplitResult) = r.test_indices

"""
    datasplit(splitter::AbstractSplitter, data) -> SplitResult

Split `data` (matrix, `DataFrame`, or vector; observations in rows) into
train and test sets whose distributions are as similar as possible, using
the method's own procedure; see the splitter types
([`SupportPointSplitter`](@ref), [`HerdingSplitter`](@ref)).
"""
function datasplit(s::SupportPointSplitter, data)
  X = preprocess(data)
  n_total = size(X, 1)
  n_small = round(Int, min(s.ratio, 1 - s.ratio) * n_total)
  0 < n_small < n_total ||
    throw(ArgumentError("ratio $(s.ratio) leaves an empty subset for $(n_total) rows"))

  kernel = resolve(s.kernel, X, s.rng)
  fitted = SupportPointSplitter(
    kernel,
    s.ratio,
    s.kappa,
    s.max_iterations,
    s.tolerance,
    s.n_threads,
    s.rng,
    s.verbose,
  )

  points, converged, iterations = support_points(
    kernel,
    X,
    n_small;
    kappa = s.kappa,
    max_iterations = s.max_iterations,
    tolerance = s.tolerance,
    n_threads = s.n_threads,
    rng = s.rng,
    verbose = s.verbose,
  )
  small = select_nearest(X, points)
  rest = setdiff(1:n_total, small)

  test, train = s.ratio <= 0.5 ? (small, rest) : (rest, small)
  return SplitResult(collect(train), collect(test), converged, iterations, fitted)
end

# `train, test = result`
Base.iterate(r::SplitResult) = (r.train_indices, :test)
Base.iterate(r::SplitResult, state::Symbol) =
  state === :test ? (r.test_indices, nothing) : nothing
Base.length(::SplitResult) = 2

function _subset_indices(r::SplitResult, subset::Symbol)
  subset === :train && return r.train_indices
  subset === :test && return r.test_indices
  throw(ArgumentError("subset must be :train or :test, got :$subset"))
end

Base.getindex(data::AbstractMatrix, r::SplitResult, subset::Symbol) =
  view(data, _subset_indices(r, subset), :)
Base.getindex(data::AbstractVector, r::SplitResult, subset::Symbol) =
  view(data, _subset_indices(r, subset))
Base.getindex(data::DataFrame, r::SplitResult, subset::Symbol) =
  view(data, _subset_indices(r, subset), :)

function Base.show(io::IO, s::SupportPointSplitter)
  print(io, "SupportPointSplitter(kernel=$(s.kernel), ratio=$(s.ratio))")
end

function Base.show(io::IO, r::SplitResult)
  print(
    io,
    "SplitResult(train=$(length(r.train_indices)), test=$(length(r.test_indices)), ",
    "converged=$(r.converged), iterations=$(r.iterations))",
  )
end
