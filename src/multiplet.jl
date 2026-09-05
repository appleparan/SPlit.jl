"""
Multiplets (Vakayil & Joseph 2022, Section 5): `k` folds that partition the
rows, each distributed like the whole data. Strategies S1 (`:sequential`)
and S2 (`:halving`) call `selectrows` repeatedly and work with every
splitter; S3 (`:single`) reads the groups of one twinning run.
"""

using DataFrames

_rows(data::AbstractMatrix, idx) = data[idx, :]
_rows(data::AbstractVector, idx) = data[idx]
_rows(data::DataFrame, idx) = data[idx, :]
_subweights(::Nothing, idx) = nothing
_subweights(weights::AbstractVector, idx) = weights[idx]

"""
    multiplet(splitter::AbstractSplitter, data, k; strategy = :sequential,
              weights = nothing, reference = nothing, reference_weights = nothing)
      -> Vector{Vector{Int}}

Partition the rows of `data` into `k` folds whose distributions each
resemble the whole data (Vakayil & Joseph 2022, Section 5). Returns `k`
sorted index vectors with sizes differing by at most one; `2 ≤ k ≤ N`.

- `:sequential` (the paper's S1, default): select `⌊N/k⌋` rows with the
  splitter, then `⌊N'/(k−1)⌋` of the remaining `N'` rows, and so on; the last
  fold is what remains. Each run preprocesses the remaining rows afresh (or,
  with `reference`, fits the preprocessing on the reference as `selectrows`
  does). The splitter's own `ratio` is ignored. Any splitter.
- `:halving` (S2): split every part into its selected half (`⌊N_part/2⌋`
  rows) and the complement, level by level, until `k` parts exist; `k`
  must be a power of two. Any splitter.
- `:single` (S3): one twinning run with `⌊N/k⌋` groups of at least `k`
  rows; fold `j` collects the `j`-th member (by neighbor rank) of every
  group, and the `N mod k`-style leftovers above rank `k` go one each to
  the first folds. [`TwinningSplitter`](@ref) only; one pass, and the
  paper measures it slightly behind S1 and S2.

`weights`, `reference`, and `reference_weights` are forwarded to
[`selectrows`](@ref) (a `TwinningSplitter` rejects them). An integer `start`
on a `TwinningSplitter` is interpreted within each run's remaining rows for
`:sequential` and `:halving` (and must be a valid position there), so
prefer `:farthest` or `:random` with those strategies.

`standardize = false` uses a numeric matrix or vector as it is (no
centering, scaling, or constant-column removal), for cosine-normalized
embeddings; a `DataFrame` then raises an `ArgumentError`.

# Examples

```julia
data = randn(MersenneTwister(1), 500, 4)
folds = multiplet(TwinningSplitter(), data, 5)
holdout = data[folds[1], :]
```

"""
function multiplet(
  s::AbstractSplitter,
  data,
  k::Integer;
  strategy::Symbol = :sequential,
  weights::Union{Nothing,AbstractVector} = nothing,
  reference = nothing,
  reference_weights::Union{Nothing,AbstractVector} = nothing,
  standardize::Bool = true,
)
  N = _nrows(data)
  2 <= k <= N || throw(ArgumentError("k must be in 2:$N, got $k"))
  k = Int(k)
  folds = if strategy === :sequential
    _multiplet_sequential(s, data, k; weights, reference, reference_weights, standardize)
  elseif strategy === :halving
    _multiplet_halving(s, data, k; weights, reference, reference_weights, standardize)
  elseif strategy === :single
    _multiplet_single(s, data, k; weights, reference, reference_weights, standardize)
  else
    throw(ArgumentError("strategy must be :sequential, :halving, or :single, got :$strategy"))
  end
  return [sort(f) for f in folds]
end

# Positions of `part` not in `sel`, in the original order

_complement(part::Vector{Int}, sel::Vector{Int}) = part[setdiff(1:length(part), sel)]

function _multiplet_sequential(
  s,
  data,
  k;
  weights,
  reference,
  reference_weights,
  standardize,
)
  remaining = collect(1:_nrows(data))
  folds = Vector{Vector{Int}}(undef, k)
  for i = 1:(k-1)
    n_i = length(remaining) ÷ (k - i + 1)
    sel = selectrows(
      s,
      _rows(data, remaining),
      n_i;
      weights = _subweights(weights, remaining),
      reference,
      reference_weights,
      standardize,
    )
    folds[i] = remaining[sel]
    remaining = _complement(remaining, sel)
  end
  folds[k] = remaining
  return folds
end

function _multiplet_halving(s, data, k; weights, reference, reference_weights, standardize)
  ispow2(k) ||
    throw(ArgumentError("strategy :halving needs k to be a power of two, got $k"))
  parts = [collect(1:_nrows(data))]
  while length(parts) < k
    next = Vector{Vector{Int}}()
    for part in parts
      sel = selectrows(
        s,
        _rows(data, part),
        length(part) ÷ 2;
        weights = _subweights(weights, part),
        reference,
        reference_weights,
        standardize,
      )
      push!(next, part[sel])
      push!(next, _complement(part, sel))
    end
    parts = next
  end
  return parts
end

function _multiplet_single(
  s::TwinningSplitter,
  data,
  k;
  weights,
  reference,
  reference_weights,
  standardize,
)
  X, _, target, _ = _prepare(s, data, weights, reference, reference_weights; standardize)
  _check_twinning_plain(weights, target)
  N = size(X, 1)
  groups = _twin_groups(X, N ÷ k, s.start, s.rng)
  folds = [Int[] for _ = 1:k]
  extra = 0
  for g in groups, (rank, row) in enumerate(g)
    if rank <= k
      push!(folds[rank], row)
    else
      extra += 1                       # fewer than k of these in total
      push!(folds[extra], row)
    end
  end
  return folds
end

_multiplet_single(::AbstractSplitter, data, k; kwargs...) =
  throw(ArgumentError("strategy :single is defined only for TwinningSplitter"))
