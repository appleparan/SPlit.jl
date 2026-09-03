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
# instead of a KDTree. Set by `benchmark/twinning_trees.jl`; see the Design
# experiments page. `typemax(Int)` means the k-d tree is always used.
const TWINNING_BRUTE_FORCE_DIMENSION = typemax(Int)

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

# Nearest alive row to `point`; `map` sends tree-local indices to data rows.
function _nearest_alive(tree, map::Vector{Int}, alive::BitVector, point)
  idx, _ = nn(tree, point, j -> !alive[map[j]])
  return map[idx]
end

# The k nearest alive rows to `point` other than row `u`, by increasing distance.
function _neighbors_alive(tree, map::Vector{Int}, alive::BitVector, point, k::Int, u::Int)
  idxs, _ = knn(tree, point, k, true, j -> (r = map[j]; !alive[r] || r == u))
  return map[idxs]
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
  map = collect(1:N)
  tree = _build_tree(Xt, map, brute_force)
  dead_in_tree = 0
  groups = Vector{Vector{Int}}(undef, n)
  far = u                                  # v_{i−1}^{r−1}; unused for i = 1
  for i = 1:n
    i > 1 && (u = _nearest_alive(tree, map, alive, view(Xt, :, far)))
    group = Vector{Int}(undef, sizes[i])
    group[1] = u
    k = sizes[i] - 1
    k > 0 && (group[2:end] = _neighbors_alive(tree, map, alive, view(Xt, :, u), k, u))
    for row in group
      alive[row] = false
    end
    dead_in_tree += length(group)
    far = group[end]
    groups[i] = group
    if i < n && 2 * dead_in_tree > length(map)
      map = findall(alive)
      tree = _build_tree(Xt, map, brute_force)
      dead_in_tree = 0
    end
  end
  return groups
end
