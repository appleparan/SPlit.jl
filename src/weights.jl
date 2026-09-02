"""
Sample-weight helpers shared by every weighted method. Weights are
per-row, non-negative, finite, with a positive sum. Two scalings are used:
`w̄` (sum one) for discrepancies and data terms, and `ŵ` (mean one) inside
the optimizers, so that uniform weights become exactly `1.0` and the
weighted arithmetic reproduces the unweighted arithmetic bit for bit.

Not part of the public API.
"""

# Validate `weights` for `N` rows: length N, all finite, none negative,
# positive sum. Returns `weights` unchanged.
function _check_weights(weights::AbstractVector, N::Int)
  length(weights) == N ||
    throw(ArgumentError("weights must have one entry per row ($N), got $(length(weights))"))
  all(w -> isfinite(w) && w >= 0, weights) ||
    throw(ArgumentError("weights must be finite and non-negative"))
  sum(weights) > 0 || throw(ArgumentError("weights must not all be zero"))
  return weights
end

# `w̄`: weights scaled to sum one, as a fresh Float64 vector.
function _normalize_weights(weights::AbstractVector, N::Int)
  _check_weights(weights, N)
  w = Vector{Float64}(weights)
  w ./= sum(w)
  return w
end

# Whether every weight is the same, i.e. the weighting is uniform.
_isuniform(w::AbstractVector) = all(==(first(w)), w)

# `ŵ`: weights scaled to mean one, as a fresh Float64 vector. Any constant
# (uniform) vector yields exactly ones, so uniform weights reproduce the
# unweighted arithmetic bit for bit.
function _mean_one_weights(weights::AbstractVector)
  sum(weights) > 0 || throw(ArgumentError("weights must not all be zero"))
  w = Vector{Float64}(weights)
  # A constant vector is exactly uniform: return exact ones rather than
  # relying on length(w) / sum(w) rounding to 1.0.
  _isuniform(w) && return fill(1.0, length(w))
  w .*= length(w) / sum(w)
  return w
end

_uniform_weights(n::Int) = fill(1.0 / n, n)

# Normalized weights for one side of a two-sample discrepancy: `nothing`
# means uniform.
_side_weights(::Nothing, n::Int) = _uniform_weights(n)
_side_weights(weights::AbstractVector, n::Int) = _normalize_weights(weights, n)

# A constant weight vector is equivalent to no weighting: normalize it to
# `nothing` so every downstream estimator (not just `Exact`/`Subsample`)
# inherits the unweighted path and reproduces its value exactly.
_uniform_as_nothing(::Nothing) = nothing
_uniform_as_nothing(w::AbstractVector) = _isuniform(w) ? nothing : w
