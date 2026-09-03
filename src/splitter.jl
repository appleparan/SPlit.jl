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
(Joseph & Vakayil 2022): the smaller side is computed as a set of support
points and mapped to data rows by sequential nearest-neighbor selection.

- `kernel`: `EnergyKernel()` (default) or `GaussianKernel(σ)`; a `:median`
  bandwidth is resolved from the data at `datasplit` time and the resolved
  kernel is stored in `result.method`.
- `ratio`: fraction of rows assigned to the test set, in (0, 1).
- `kappa`: absolute per-iteration subsample size for stochastic optimization;
  `nothing` uses all rows every iteration. Stochastic mode runs only when
  `kappa` is below the number of rows of the target (the data, or the
  reference when one is given).
- `tolerance`: convergence when the largest squared displacement of any
  support point in one iteration is below this value. In stochastic mode
  the running-average weight decays with the iteration count, so
  convergence there partly reflects that step-size decay rather than the
  objective flattening out. For `GaussianKernel`, convergence never fires
  before the second iteration, and also triggers when the relative
  objective decrease falls below an internal `rtol = 1e-8` (not exposed
  here).
- `rng`: source of all randomness (initialization, jitter, stochastic
  subsampling); pass a seeded RNG for reproducible splits.
- `verbose`: print per-iteration progress.

In high dimension the optimized points may move less than the spacing
between rows, in which case the selection is the initial random sample;
prefer `HerdingSplitter` there (see the Benchmarks page).
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
optimizer's convergence. `selected` is the side (`:test` or `:train`) that
holds the rows the splitter chose; the other side is the complement.
"""
struct SplitResult{M<:AbstractSplitter}
  train_indices::Vector{Int}
  test_indices::Vector{Int}
  converged::Bool
  iterations::Int
  method::M
  selected::Symbol
end

SplitResult(train, test, converged, iterations, method) = SplitResult(
  train,
  test,
  converged,
  iterations,
  method,
  length(test) <= length(train) ? :test : :train,
)

"""
    train_indices(result::SplitResult) -> Vector{Int}
"""
train_indices(r::SplitResult) = r.train_indices

"""
    test_indices(result::SplitResult) -> Vector{Int}
"""
test_indices(r::SplitResult) = r.test_indices

_with_kernel(s::SupportPointSplitter, kernel) = SupportPointSplitter(
  kernel,
  s.ratio,
  s.kappa,
  s.max_iterations,
  s.tolerance,
  s.n_threads,
  s.rng,
  s.verbose,
)

# Preprocess and resolve the kernel for the data-as-target case (weights)
# or the reference case. Returns the encoded data, the resolved kernel, the
# encoded target (or nothing), and the target weights.
function _prepare(s::AbstractSplitter, data, weights, reference, reference_weights)
  if reference === nothing
    reference_weights === nothing ||
      throw(ArgumentError("reference_weights needs a reference"))
    X = preprocess(data, weights)
    return X, resolve(s.kernel, X, s.rng, weights), nothing, nothing
  end
  weights === nothing || throw(
    ArgumentError(
      "with a reference, weight the reference (reference_weights), not the data",
    ),
  )
  _nrows(reference) >= 1 || throw(ArgumentError("reference must have at least one row"))
  prep = fit_preprocessor(reference; weights = reference_weights, extra = data)
  R = apply_preprocessor(prep, reference)
  X = apply_preprocessor(prep, data)
  return X, resolve(s.kernel, R, s.rng, reference_weights), R, reference_weights
end

function _select_rows(
  s::SupportPointSplitter,
  kernel,
  X,
  n;
  weights,
  target,
  target_weights,
)
  points, converged, iterations = support_points(
    kernel,
    X,
    n;
    kappa = s.kappa,
    max_iterations = s.max_iterations,
    tolerance = s.tolerance,
    n_threads = s.n_threads,
    rng = s.rng,
    verbose = s.verbose,
    weights,
    target,
    target_weights,
  )
  return select_nearest(X, points), converged, iterations
end

function _select(
  s::AbstractSplitter,
  data,
  n::Integer;
  weights = nothing,
  reference = nothing,
  reference_weights = nothing,
)
  X, kernel, target, target_weights =
    _prepare(s, data, weights, reference, reference_weights)
  N = size(X, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$(N), got $n"))
  indices, converged, iterations =
    _select_rows(s, kernel, X, Int(n); weights, target, target_weights)
  return indices, converged, iterations, _with_kernel(s, kernel)
end

"""
    selectrows(splitter::AbstractSplitter, data, n; weights = nothing,
           reference = nothing, reference_weights = nothing) -> Vector{Int}

Indices of the `n` rows of `data` the splitter chooses, in selection order
(support-point order for `SupportPointSplitter`, greedy order for
`HerdingSplitter`), without building a train/test partition. The chosen
rows approximate the data's own distribution (weighted by `weights`) or,
when `reference` is given, the distribution of `reference` (weighted by
`reference_weights`): preprocessing is then fit on `reference` and applied
to both, candidates stay the rows of `data`, and `weights` may not be
given. Convergence diagnostics are reported by [`datasplit`](@ref).
"""
function selectrows(
  s::AbstractSplitter,
  data,
  n::Integer;
  weights::Union{Nothing,AbstractVector} = nothing,
  reference = nothing,
  reference_weights::Union{Nothing,AbstractVector} = nothing,
)
  return _select(s, data, n; weights, reference, reference_weights)[1]
end

_nrows(data::AbstractMatrix) = size(data, 1)
_nrows(data::AbstractVector) = length(data)
_nrows(data::DataFrame) = nrow(data)

"""
    datasplit(splitter::AbstractSplitter, data) -> SplitResult

Split `data` (matrix, `DataFrame`, or vector; observations in rows) into
train and test sets whose distributions are as similar as possible, using
the method's own procedure; see the splitter types
([`SupportPointSplitter`](@ref), [`HerdingSplitter`](@ref)).

`weights` (one non-negative entry per row; `nothing` for uniform) makes the
split target the weighted empirical distribution `Σ w̄ᵢ δ(xᵢ)`: the smaller
subset is chosen to approximate it, preprocessing standardizes with the
weighted mean and variance, and a `:median` bandwidth is resolved from rows
drawn in proportion to the weights — this only changes the resolved
bandwidth for datasets above 1000 rows; below that every row enters the
median and the weights do not change it. The train/test labeling rule is
unchanged. Weights proportional to duplication counts are equivalent to
duplicating rows, up to the common column rescaling of the weighted
standardization, which changes nothing under `EnergyKernel` or a `:median`
bandwidth but does matter for a fixed numeric Gaussian bandwidth.

`reference` (same kind and columns as `data`; optionally weighted by
`reference_weights`) makes the chosen side approximate the distribution of
`reference` instead of the data: preprocessing is fit on `reference` and
applied to both sets, a `:median` bandwidth is resolved on the encoded
reference, and candidates remain the rows of `data`. `weights` cannot be
combined with `reference`. The train/test labeling rule is unchanged;
`result.selected` names the side that holds the chosen rows. See
[`selectrows`](@ref) for the indices alone.
"""
function datasplit(
  s::AbstractSplitter,
  data;
  weights::Union{Nothing,AbstractVector} = nothing,
  reference = nothing,
  reference_weights::Union{Nothing,AbstractVector} = nothing,
)
  n_total = _nrows(data)
  n_small = round(Int, min(s.ratio, 1 - s.ratio) * n_total)
  0 < n_small < n_total ||
    throw(ArgumentError("ratio $(s.ratio) leaves an empty subset for $(n_total) rows"))
  small, converged, iterations, fitted =
    _select(s, data, n_small; weights, reference, reference_weights)
  rest = setdiff(1:n_total, small)
  test, train = s.ratio <= 0.5 ? (small, rest) : (rest, small)
  selected = s.ratio <= 0.5 ? :test : :train
  return SplitResult(collect(train), collect(test), converged, iterations, fitted, selected)
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
    "converged=$(r.converged), iterations=$(r.iterations), selected=$(r.selected))",
  )
end
