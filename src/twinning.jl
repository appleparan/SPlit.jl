"""
Twinning (Vakayil & Joseph 2022, Algorithm 1): sequential nearest-neighbor
grouping of the standardized rows. `n` disjoint groups cover the data; each
group is a row `u_i` plus its nearest ungrouped neighbors, and the next `u`
is the ungrouped row nearest to the farthest member of the previous group.
Selecting one row per group gives the twin. Serial by design.
"""

using NearestNeighbors
using Random

# Dimension at or above which the twinning search structure is a BruteTree
# instead of a KDTree. Set by `benchmark/twinning_trees.jl`: on
# standard-normal data at N = 10,000, brute force is 2.03x faster at p = 50
# (1.29x at p = 200, 2.00x at p = 768); see the Design experiments page.
# `typemax(Int)` would mean the k-d tree is always used.
const TWINNING_BRUTE_FORCE_DIMENSION = 50

# Sizes of the n groups: r = N ÷ n rows each, with the N − r·n leftover rows
# spread as groups of r + 1 evenly along the chain (spec, decision 1). All
# groups have r rows when N is a multiple of n, which is the paper's case.
function _group_sizes(N::Int, n::Int)
  r, extra = divrem(N, n)
  sizes = fill(r, n)
  for i = 1:n
    (i * extra) ÷ n > ((i - 1) * extra) ÷ n && (sizes[i] = r + 1)
  end
  return sizes
end

# Starting row u₁. `:farthest` is the row farthest from the origin, which is
# the centroid of standardized data (paper, Section 4); ties go to the
# lowest index and no rng is consumed. `:random` draws a row with `rng`
# (paper, Section 5). An integer is used as given (Algorithm 1 input).
function _start_row(start, X::Matrix{Float64}, rng::AbstractRNG)
  N = size(X, 1)
  start === :farthest && return argmax(vec(sum(abs2, X; dims = 2)))
  start === :random && return rand(rng, 1:N)
  start isa Integer ||
    throw(ArgumentError("start must be :farthest, :random, or a row index, got $start"))
  1 <= start <= N || throw(ArgumentError("start must be a row index in 1:$N, got $start"))
  return Int(start)
end

_build_tree(Xt::Matrix{Float64}, rows::Vector{Int}, brute_force::Bool) =
  brute_force ? BruteTree(Xt[:, rows]) : KDTree(Xt[:, rows])

# Nearest alive row to `point`; `rows` sends tree-local indices to data rows.
function _nearest_alive(tree, rows::Vector{Int}, alive::BitVector, point)
  idx, _ = nn(tree, point, j -> !alive[rows[j]])
  return rows[idx]
end

# The k nearest alive rows to `point` other than row `u`, by increasing distance.
function _neighbors_alive(tree, rows::Vector{Int}, alive::BitVector, point, k::Int, u::Int)
  idxs, _ = knn(tree, point, k, true, j -> (r = rows[j]; !alive[r] || r == u))
  return rows[idxs]
end

"""
    _twin_groups(X, n, start, rng; brute_force) -> Vector{Vector{Int}}

The `n` groups of Algorithm 1 on the standardized rows of `X`, in formation
order; each group lists `u_i` first and then its neighbors by increasing
distance to `u_i`. Grouped rows are masked in the search tree; when more
than half of the tree's rows are masked the tree is rebuilt on the alive
rows (an implementation detail, not from the paper; total rebuild work is
`O(N log N)`). `brute_force` selects a `BruteTree` over a `KDTree`.
"""
function _twin_groups(
  X::Matrix{Float64},
  n::Int,
  start,
  rng::AbstractRNG;
  brute_force::Bool = size(X, 2) >= TWINNING_BRUTE_FORCE_DIMENSION,
)
  N = size(X, 1)
  0 < n <= N || throw(ArgumentError("n must be in 1:$N, got $n"))
  u = _start_row(start, X, rng)
  sizes = _group_sizes(N, n)
  Xt = permutedims(X)                      # rows as contiguous columns
  alive = trues(N)
  rows = collect(1:N)
  tree = _build_tree(Xt, rows, brute_force)
  dead_in_tree = 0
  groups = Vector{Vector{Int}}(undef, n)
  far = u                                  # v_{i−1}^{r−1}; unused for i = 1
  for i = 1:n
    i > 1 && (u = _nearest_alive(tree, rows, alive, view(Xt, :, far)))
    group = Vector{Int}(undef, sizes[i])
    group[1] = u
    k = sizes[i] - 1
    k > 0 && (group[2:end] = _neighbors_alive(tree, rows, alive, view(Xt, :, u), k, u))
    for row in group
      alive[row] = false
    end
    dead_in_tree += length(group)
    far = group[end]
    groups[i] = group
    if i < n && 2 * dead_in_tree > length(rows)
      rows = findall(alive)
      tree = _build_tree(Xt, rows, brute_force)
      dead_in_tree = 0
    end
  end
  return groups
end

"""
    TwinningSplitter(; ratio = 0.2, start = :farthest, rng = Random.default_rng())

Split by twinning (Vakayil & Joseph 2022): the standardized rows are
covered by `n` disjoint groups, each a row `u_i` and its nearest ungrouped
neighbors, formed as a chain where `u_{i+1}` is the ungrouped row nearest
to the farthest member of group `i`; the smaller side is `{u_1, …, u_n}`.
The objective is the energy distance (the paper's Proposition 1 ties it to
the SPlit objective), so `kernel` is a fixed `EnergyKernel()`. No optimizer,
no `kappa`, serial; average cost `O(pN log N)`.

- `ratio`: fraction of rows assigned to the test set, in (0, 1).
- `start`: the starting row `u_1`: `:farthest` (default) is the row farthest
  from the centroid of the standardized data and consumes no randomness,
  `:random` draws it with `rng`, an integer names the row.
- `rng`: used only by `start = :random`.

`SplitResult.converged` is always `true` and `iterations` is the number of
groups formed. `weights` and `reference` are not defined for twinning and
raise an `ArgumentError`.

# Differences from the paper

The paper assumes `1/ratio = r` is an integer and takes `n = ⌈N/r⌉`, with
the last group absorbing the remainder. Here `n` follows the generic rule
of [`datasplit`](@ref) (or the caller's `n` in [`selectrows`](@ref)) and
`r = ⌊N/n⌋`, the `N − rn` leftover rows forming groups of `r + 1` spread
evenly along the chain; when `N = rn` the two agree exactly. Grouped rows
are masked in a k-d tree that is rebuilt on the remaining rows once more
than half of its rows are masked (the paper masks without rebuilding).

# Examples

```julia
data = randn(MersenneTwister(1), 1_000, 5)
result = datasplit(TwinningSplitter(), data)          # deterministic
folds = multiplet(TwinningSplitter(), data, 5)        # 5 distribution-balanced folds
```

"""
struct TwinningSplitter{R<:AbstractRNG} <: AbstractSplitter
  kernel::EnergyKernel
  ratio::Float64
  start::Union{Symbol,Int}
  rng::R
end

function TwinningSplitter(;
  ratio::Real = 0.2,
  start::Union{Symbol,Integer} = :farthest,
  rng::AbstractRNG = Random.default_rng(),
)
  ratio = Float64(ratio)
  0 < ratio < 1 || throw(ArgumentError("ratio must be in (0, 1), got $ratio"))
  if start isa Symbol
    start in (:farthest, :random) ||
      throw(ArgumentError("start must be :farthest, :random, or a row index, got :$start"))
  else
    start >= 1 || throw(ArgumentError("start must be a positive row index, got $start"))
    start = Int(start)
  end
  return TwinningSplitter(EnergyKernel(), ratio, start, rng)
end

_with_kernel(s::TwinningSplitter, kernel) = s

function _check_twinning_plain(weights, target)
  (weights === nothing && target === nothing) || throw(
    ArgumentError(
      "TwinningSplitter has no weighted or reference form; the paper defines it on the data alone",
    ),
  )
  return nothing
end

function _select_rows(s::TwinningSplitter, kernel, X, n; weights, target, target_weights)
  _check_twinning_plain(weights, target)
  groups = _twin_groups(X, n, s.start, s.rng)
  return first.(groups), true, n
end

function Base.show(io::IO, s::TwinningSplitter)
  start = s.start isa Symbol ? ":$(s.start)" : string(s.start)
  print(io, "TwinningSplitter(ratio=$(s.ratio), start=$start)")
end
