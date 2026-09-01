"""
Internal data preprocessing: categorical encoding (Helmert contrasts),
constant-column removal, and standardization to mean 0 / variance 1.

Not part of the public API.
"""

using DataFrames
using CategoricalArrays
using Statistics

"""
    helmert_matrix(nlevels::Int) -> Matrix{Float64}

Helmert contrast matrix of size `(nlevels, nlevels - 1)`. Column `j` assigns
`1/j` to the first `j` levels and `-1` to level `j + 1`.
"""
function helmert_matrix(nlevels::Int)
  nlevels <= 1 && return zeros(nlevels, 0)
  H = zeros(nlevels, nlevels - 1)
  for j = 1:(nlevels-1)
    H[1:j, j] .= 1.0 / j
    H[j+1, j] = -1.0
  end
  return H
end

_is_constant(col) = all(x -> isequal(x, first(col)), col)

function _standardize!(M::Matrix{Float64})
  for j in axes(M, 2)
    μ = mean(view(M, :, j))
    σ = std(view(M, :, j))
    @views M[:, j] .= (M[:, j] .- μ) ./ σ
  end
  return M
end

"""
    preprocess(data) -> Matrix{Float64}

Validate and transform `data` for splitting: reject missing values, encode
categorical columns with Helmert contrasts, drop constant columns, and
standardize every remaining column. Accepts `AbstractMatrix`, `DataFrame`,
and `AbstractVector` inputs.
"""
function preprocess(data::AbstractMatrix)
  any(ismissing, data) && throw(ArgumentError("Dataset contains missing value(s)."))
  all(x -> x isa Number, data) ||
    throw(ArgumentError("Matrix input must contain only numeric values."))
  keep = [!_is_constant(view(data, :, j)) for j in axes(data, 2)]
  any(keep) || throw(ArgumentError("All columns are constant."))
  return _standardize!(Float64.(data[:, keep]))
end

preprocess(data::AbstractVector) = preprocess(reshape(collect(data), :, 1))

_is_categorical(col) =
  Base.nonmissingtype(eltype(col)) <: AbstractString || col isa CategoricalVector

# Canonical level order, independent of row order: declaration order (filtered
# to levels actually present) for a `CategoricalVector`, sorted order for
# plain string columns.
function _canonical_levels(col)
  col isa CategoricalVector && return filter(in(unique(col)), levels(col))
  return sort(unique(col))
end

function preprocess(data::DataFrame)
  for col in eachcol(data)
    any(ismissing, col) && throw(ArgumentError("Dataset contains missing value(s)."))
  end

  columns = Vector{Vector{Float64}}()
  for name in names(data)
    col = data[!, name]
    if _is_categorical(col)
      levels_ = _canonical_levels(col)
      length(levels_) <= 1 && continue
      H = helmert_matrix(length(levels_))
      index = Dict(l => i for (i, l) in enumerate(levels_))
      for j in axes(H, 2)
        push!(columns, [H[index[v], j] for v in col])
      end
    elseif Base.nonmissingtype(eltype(col)) <: Number
      _is_constant(col) && continue
      push!(columns, Float64.(col))
    else
      throw(ArgumentError("Unsupported column type in column: $(name)"))
    end
  end

  isempty(columns) && throw(ArgumentError("All columns are constant."))
  # hcat(columns...) (not reduce) so a single column still yields an n×1 Matrix
  return _standardize!(hcat(columns...))
end
