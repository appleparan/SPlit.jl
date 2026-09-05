# Helpers for flattening time series into fixed-length windows so a
# distribution-preserving selector can be applied to them. Base + Random +
# Statistics + LinearAlgebra only, so both `examples/time_series_windows.jl`
# and `test/test_time_series_windows.jl` can `include` this file without
# picking up the example project's extra dependencies.

using Random
using Statistics
using LinearAlgebra

"""
    window_count(N, L, stride) -> Int

Number of length-`L` windows a stride-`stride` sliding window produces over
`N` observations: `(N - L) ÷ stride + 1`, or `0` when `N < L` (not an
error — a trailing partial window is dropped rather than padded). `L` and
`stride` must be positive.
"""
function window_count(N::Integer, L::Integer, stride::Integer)
  L > 0 || throw(ArgumentError("L must be positive, got $L"))
  stride > 0 || throw(ArgumentError("stride must be positive, got $stride"))
  N < L && return 0
  return (N - L) ÷ stride + 1
end

"""
    windows(X, L; stride = L) -> (Z, starts)

Flatten `X` (`N × p`, observations in rows) into overlapping or
non-overlapping windows of length `L`. Row `i` of `Z` (`M × L*p`,
`M = window_count(N, L, stride)`) is `vec(X[s:s+L-1, :])` — variable-major:
all `L` offsets of variable 1, then all `L` offsets of variable 2, and so
on — where `s = starts[i]`. `N < L` returns a `0 × L*p` matrix and an empty
`starts`, not an error.
"""
function windows(X::AbstractMatrix, L::Integer; stride::Integer = L)
  N, p = size(X)
  M = window_count(N, L, stride)
  Z = Matrix{Float64}(undef, M, L * p)
  starts = Vector{Int}(undef, M)
  for i = 1:M
    s = (i - 1) * stride + 1
    starts[i] = s
    Z[i, :] = vec(X[s:(s+L-1), :])
  end
  return Z, starts
end

"""
    recover_window(X, start, L) -> Matrix

The original `L × p` slice `X[start:start+L-1, :]` a window came from.
`reshape(Z[i, :], L, p) == recover_window(X, starts[i], L)` for the `Z`,
`starts` returned by [`windows`](@ref).
"""
recover_window(X::AbstractMatrix, start::Integer, L::Integer) = X[start:(start+L-1), :]

"""
    standardize_by_variable(Z, L, p; fit = Z) -> Matrix{Float64}

Standardize the flattened windows `Z` (`M × L*p`, variable-major as
returned by [`windows`](@ref)) one variable at a time: for variable `v`
(columns `(v-1)*L+1 : v*L`), subtract the mean and divide by the standard
deviation computed over every offset and every window of `fit` — so all
`L` offsets of a variable share one scale, since they are the same
physical quantity at different lags. `fit` defaults to `Z` itself; pass the
training block's windows to fit on a chronological split and apply the
same shift and scale to held-out windows. A variable constant on `fit`
(zero standard deviation) uses scale `1.0` instead of dividing by zero:
the block becomes all zeros after centering, not `NaN`.
"""
function standardize_by_variable(
  Z::AbstractMatrix,
  L::Integer,
  p::Integer;
  fit::AbstractMatrix = Z,
)
  size(Z, 2) == L * p ||
    throw(ArgumentError("Z must have L*p = $(L*p) columns, got $(size(Z, 2))"))
  Zs = Matrix{Float64}(undef, size(Z, 1), size(Z, 2))
  for v = 1:p
    cols = ((v-1)*L+1):(v*L)
    block = view(fit, :, cols)
    m = mean(block)
    s = std(block)
    s = s == 0.0 ? 1.0 : s
    Zs[:, cols] = (view(Z, :, cols) .- m) ./ s
  end
  return Zs
end

"""
    lag1_autocorrelation(z, L, p) -> Float64

Mean, over the `p` variables, of the lag-1 sample autocorrelation of one
flattened window `z` (length `L*p`, variable-major as in [`windows`](@ref)):
for variable `v`'s `L` values `x`, `sum((x[t]-x̄)*(x[t+1]-x̄)) /
sum((x[t]-x̄)^2)`. A constant variable (zero denominator) contributes `0.0`
rather than `NaN`.
"""
function lag1_autocorrelation(z::AbstractVector, L::Integer, p::Integer)
  total = 0.0
  for v = 1:p
    cols = ((v-1)*L+1):(v*L)
    x = view(z, cols)
    xm = mean(x)
    d = x .- xm
    denom = sum(abs2, d)
    total += denom == 0.0 ? 0.0 : sum(d[1:(end-1)] .* d[2:end]) / denom
  end
  return total / p
end

"""
    two_regime_series(rng; M, L, p = 3, share_a = 0.7, stay_a = 0.94,
                       stay_b = 0.10, mu = (1.0, 0.7, 1.3), sigma = 0.4)
    -> (X, labels)

Synthetic series of `M` non-overlapping windows of length `L` (`N = M*L`
rows, `p` columns), built so each window's point-level mean and variance
per variable match across regimes while its temporal dependence differs.
Each window independently draws a regime (`:A` with probability
`share_a`, else `:B`) and a two-state Markov chain `s_t ∈ {-1, +1}`
starting at `±1` with equal probability, with stay probability `stay_a`
(persistent, mean run length `1/(1-stay_a)`) for `:A` and `stay_b`
(alternating) for `:B`. Variable `v`'s value at offset `t` is
`s_t * a * mu[v] + sigma * eps_t`, `eps_t ~ N(0,1)` drawn independently per
variable, and `a ~ Uniform(0.8, 1.2)` a per-window amplitude factor shared
across variables and offsets. `labels` names the regime of each window
(never seen by a selector). All randomness is drawn from `rng`. `M`, `L`,
and `p` must be positive, and `mu` must have at least `p` entries.
"""
function two_regime_series(
  rng::AbstractRNG;
  M::Integer,
  L::Integer,
  p::Integer = 3,
  share_a::Real = 0.7,
  stay_a::Real = 0.94,
  stay_b::Real = 0.10,
  mu::NTuple = (1.0, 0.7, 1.3),
  sigma::Real = 0.4,
)
  M > 0 || throw(ArgumentError("M must be positive, got $M"))
  L > 0 || throw(ArgumentError("L must be positive, got $L"))
  p > 0 || throw(ArgumentError("p must be positive, got $p"))
  length(mu) >= p ||
    throw(ArgumentError("mu must have at least p = $p entries, got $(length(mu))"))
  N = M * L
  X = Matrix{Float64}(undef, N, p)
  labels = Vector{Symbol}(undef, M)
  for m = 1:M
    is_a = rand(rng) < share_a
    labels[m] = is_a ? :A : :B
    stay = is_a ? stay_a : stay_b
    a = 0.8 + 0.4 * rand(rng)               # Uniform(0.8, 1.2)
    s = rand(rng, Bool) ? 1.0 : -1.0
    row0 = (m - 1) * L
    for t = 1:L
      t > 1 && rand(rng) >= stay && (s = -s)
      for v = 1:p
        X[row0+t, v] = s * a * mu[v] + sigma * randn(rng)
      end
    end
  end
  return X, labels
end
