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

_is_categorical(col) =
  Base.nonmissingtype(eltype(col)) <: AbstractString || col isa CategoricalVector

# Canonical level order, independent of row order: declaration order (filtered
# to levels actually present) for a `CategoricalVector`, sorted order for
# plain string columns.
function _canonical_levels(col)
  col isa CategoricalVector && return filter(in(unique(col)), levels(col))
  return sort(unique(col))
end

# Per input-column encoding rule learned at fit time.
abstract type ColumnSpec end
struct NumericColumn <: ColumnSpec end
struct CategoricalColumn <: ColumnSpec
  levels::Vector{String}
end

"""
    Preprocessor

Fitted preprocessing: per-column encoding rules, which encoded columns are
kept (those not constant on both the fit set and, when given, the extra
set), and the mean and scale of every kept column. Built by
[`fit_preprocessor`](@ref), applied by [`apply_preprocessor`](@ref).
Internal.
"""
struct Preprocessor
  names::Union{Nothing,Vector{String}}
  specs::Vector{ColumnSpec}
  keep::Vector{Bool}
  μ::Vector{Float64}
  σ::Vector{Float64}
end

_check_no_missing(data::AbstractMatrix) =
  any(ismissing, data) && throw(ArgumentError("Dataset contains missing value(s)."))
function _check_no_missing(data::DataFrame)
  for col in eachcol(data)
    any(ismissing, col) && throw(ArgumentError("Dataset contains missing value(s)."))
  end
end

# Union of the canonical levels of `col` and, when given, `extra_col`:
# `col`'s canonical order first, then the levels only `extra_col` has, in
# `extra_col`'s canonical order.
function _union_levels(col, extra_col)
  levels_ = string.(_canonical_levels(col))
  extra_col === nothing && return levels_
  for l in string.(_canonical_levels(extra_col))
    l in levels_ || push!(levels_, l)
  end
  return levels_
end

# Column specs from the fit set (and the extra set's levels).
function _column_specs(data::AbstractMatrix, extra)
  _check_no_missing(data)
  all(x -> x isa Number, data) ||
    throw(ArgumentError("Matrix input must contain only numeric values."))
  if extra !== nothing
    extra isa AbstractMatrix ||
      throw(ArgumentError("reference and data must both be matrices or both DataFrames"))
    size(extra, 2) == size(data, 2) ||
      throw(ArgumentError("reference and data must have the same number of columns"))
    _check_no_missing(extra)
    all(x -> x isa Number, extra) ||
      throw(ArgumentError("Matrix input must contain only numeric values."))
  end
  return nothing, ColumnSpec[NumericColumn() for _ in axes(data, 2)]
end

function _column_specs(data::DataFrame, extra)
  _check_no_missing(data)
  if extra !== nothing
    extra isa DataFrame ||
      throw(ArgumentError("reference and data must both be matrices or both DataFrames"))
    names(extra) == names(data) || throw(
      ArgumentError("reference and data must have the same column names in the same order"),
    )
    _check_no_missing(extra)
  end
  specs = ColumnSpec[]
  for name in names(data)
    col = data[!, name]
    extra_col = extra === nothing ? nothing : extra[!, name]
    if _is_categorical(col)
      extra_col === nothing ||
        _is_categorical(extra_col) ||
        throw(
          ArgumentError(
            "column $(name) is categorical in one set and numeric in the other",
          ),
        )
      push!(specs, CategoricalColumn(_union_levels(col, extra_col)))
    elseif Base.nonmissingtype(eltype(col)) <: Number
      extra_col === nothing ||
        Base.nonmissingtype(eltype(extra_col)) <: Number ||
        throw(
          ArgumentError(
            "column $(name) is categorical in one set and numeric in the other",
          ),
        )
      push!(specs, NumericColumn())
    else
      throw(ArgumentError("Unsupported column type in column: $(name)"))
    end
  end
  return String.(names(data)), specs
end

# Encode `data` with fixed specs into the full (unfiltered) column matrix.
function _encode(names_::Nothing, specs::Vector{ColumnSpec}, data::AbstractMatrix)
  _check_no_missing(data)
  size(data, 2) == length(specs) ||
    throw(ArgumentError("expected $(length(specs)) columns, got $(size(data, 2))"))
  all(x -> x isa Number, data) ||
    throw(ArgumentError("Matrix input must contain only numeric values."))
  return Float64.(data)
end
_encode(::Vector{String}, ::Vector{ColumnSpec}, ::AbstractMatrix) =
  throw(ArgumentError("the preprocessor was fit on a DataFrame; pass a DataFrame"))
_encode(::Nothing, ::Vector{ColumnSpec}, ::DataFrame) =
  throw(ArgumentError("the preprocessor was fit on a matrix; pass a matrix"))

function _encode(names_::Vector{String}, specs::Vector{ColumnSpec}, data::DataFrame)
  _check_no_missing(data)
  names(data) == names_ ||
    throw(ArgumentError("expected columns $(names_), got $(names(data))"))
  columns = Vector{Vector{Float64}}()
  for (name, spec) in zip(names_, specs)
    col = data[!, name]
    if spec isa CategoricalColumn
      _is_categorical(col) || throw(ArgumentError("column $(name) must be categorical"))
      index = Dict(l => i for (i, l) in enumerate(spec.levels))
      H = helmert_matrix(length(spec.levels))
      rows = map(col) do v
        get(index, string(v)) do
          throw(ArgumentError("unknown level $(repr(v)) in column $(name)"))
        end
      end
      for j in axes(H, 2)
        push!(columns, [H[r, j] for r in rows])
      end
    else
      Base.nonmissingtype(eltype(col)) <: Number ||
        throw(ArgumentError("column $(name) must be numeric"))
      push!(columns, Float64.(col))
    end
  end
  isempty(columns) && return zeros(nrow(data), 0)
  # hcat(columns...) (not reduce) so a single column still yields an n×1 Matrix
  return hcat(columns...)
end

# Vectors are single-column matrices.
_as_matrix(x::AbstractVector) = reshape(collect(x), :, 1)
_as_matrix(x) = x

"""
    fit_preprocessor(data; weights = nothing, extra = nothing) -> Preprocessor

Learn the preprocessing on `data`: categorical columns are Helmert-encoded
over the canonical-order union of their levels in `data` and in `extra`
(the set the preprocessor will also be applied to). An encoded column
constant on both `data` and `extra` (or constant on `data` when `extra` is
not given) carries no information and is dropped. A column constant on
`data` but varying on `extra` is kept instead of dropped: it is centered at
`data`'s constant value and scaled by `extra`'s own (unweighted) spread, so
it standardizes `data`'s rows to exactly 0 and penalizes `extra` rows away
from that value. Every other kept column gets the mean and scale of `data`
(weighted forms when `weights` is given, as in [`preprocess`](@ref));
`weights` never applies to `extra`. With `extra === nothing`, this is
exactly the previous behavior and `preprocess` stays bit-identical.
Internal.
"""
function fit_preprocessor(data; weights = nothing, extra = nothing)
  data = _as_matrix(data)
  extra = _as_matrix(extra)
  names_, specs = _column_specs(data, extra)
  M = _encode(names_, specs, data)
  size(M, 1) >= 1 || throw(ArgumentError("the fit set must have at least one row"))
  keep_fit = [!_is_constant(view(M, :, j)) for j in axes(M, 2)]
  E = extra === nothing ? nothing : _encode(names_, specs, extra)
  keep = if E === nothing
    keep_fit
  else
    keep_extra = [!_is_constant(view(E, :, j)) for j in axes(E, 2)]
    keep_fit .| keep_extra
  end
  any(keep) || throw(ArgumentError("All columns are constant."))
  kept_cols = findall(keep)
  K = M[:, keep]
  N = size(K, 1)
  weights === nothing || _check_weights(weights, N)
  w = _uniform_as_nothing(weights)
  μ = Vector{Float64}(undef, size(K, 2))
  σ = Vector{Float64}(undef, size(K, 2))
  if w === nothing
    for (k, j) in enumerate(kept_cols)
      if keep_fit[j]
        μ[k] = mean(view(K, :, k))
        σ[k] = std(view(K, :, k))
      else
        μ[k] = M[1, j]
        σ[k] = std(view(E, :, j))
      end
    end
  else
    wn = _normalize_weights(w, N)
    correction = 1 - sum(abs2, wn)
    correction > 0 || throw(ArgumentError("weights must be positive on at least two rows"))
    for (k, j) in enumerate(kept_cols)
      if keep_fit[j]
        col = view(K, :, k)
        μ[k] = sum(wn .* col)
        σ[k] = sqrt(sum(wn .* (col .- μ[k]) .^ 2) / correction)
        σ[k] > 0 || throw(
          ArgumentError(
            "column $j is constant on the rows with positive weight; drop it or give those rows weight",
          ),
        )
      else
        μ[k] = M[1, j]
        σ[k] = std(view(E, :, j))
      end
    end
  end
  return Preprocessor(names_, specs, keep, μ, σ)
end

"""
    apply_preprocessor(prep::Preprocessor, data) -> Matrix{Float64}

Encode `data` with the rules of `prep` and standardize every kept column
with `prep`'s mean and scale. A categorical level `prep` has not seen, a
column-count or column-name mismatch, and a numeric/categorical kind
mismatch are `ArgumentError`s. Internal.
"""
function apply_preprocessor(prep::Preprocessor, data)
  M = _encode(prep.names, prep.specs, _as_matrix(data))[:, prep.keep]
  for j in axes(M, 2)
    @views M[:, j] .= (M[:, j] .- prep.μ[j]) ./ prep.σ[j]
  end
  return M
end

"""
    preprocess(data) -> Matrix{Float64}
    preprocess(data, weights) -> Matrix{Float64}

Validate and transform `data` for splitting: reject missing values, encode
categorical columns with Helmert contrasts, drop constant columns, and
standardize every remaining column. Accepts `AbstractMatrix`, `DataFrame`,
and `AbstractVector` inputs. Equivalent to fitting a [`Preprocessor`](@ref)
on `data` and applying it to `data`.

With `weights` (one non-negative entry per row), standardization uses the
weighted mean `μⱼ = Σ w̄ᵢ xᵢⱼ` and the unbiased weighted variance
`σⱼ² = Σ w̄ᵢ (xᵢⱼ − μⱼ)² / (1 − Σ w̄ᵢ²)` with `w̄` the weights scaled to sum
one, which reduces to the `n − 1` denominator of `std` for uniform weights;
the encoding steps are the same. `weights = nothing` is the unweighted
method. A constant weight vector is treated as `nothing`, so uniform
weights take the unweighted path and reproduce it exactly.
"""
preprocess(data) = apply_preprocessor(fit_preprocessor(data), data)
preprocess(data, ::Nothing) = preprocess(data)
preprocess(data, weights::AbstractVector) =
  apply_preprocessor(fit_preprocessor(data; weights), data)
