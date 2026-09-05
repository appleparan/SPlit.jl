"""
Sequential nearest-neighbor selection: each support point, in order, claims
its nearest not-yet-selected data row (Joseph & Vakayil 2022). Served by a
k-d tree (`search = :kdtree`; doubles `k` and retries when every returned
neighbor is already claimed) or, at or above `NEAREST_BRUTE_FORCE_DIMENSION`
columns, by [`MatrixSearch`](@ref) (`search = :matrix`; one pass per point,
skipping claimed rows) — a NearestNeighbors tree specializes its search code
on `SVector{p, Float64}` and stops compiling in practical time once `p`
reaches a few thousand, while `MatrixSearch`'s plain `Matrix{Float64}`
compiles once regardless of width (issue #72).
"""

using NearestNeighbors

# Dimension at or above which select_nearest defaults to MatrixSearch instead
# of a KDTree. Set by `benchmark/brute_force.jl` (2026-09-05, N = 10,000 rows,
# 2,000 query points near the rows): the matrix search matches the k-d tree
# at p = 200 (0.37 s vs 0.41 s) and is 3x faster at p = 768, while the k-d
# tree is 5-10x faster at p <= 50 for N = 100,000; the k-d tree also pays a
# width-specific first-call compilation (1.4 s at p = 200, 12 s at p = 768).
# See the Design experiments page.
const NEAREST_BRUTE_FORCE_DIMENSION = 200

_nearest_search(p::Int) = p >= NEAREST_BRUTE_FORCE_DIMENSION ? :matrix : :kdtree

"""
    MatrixSearch(Xt)

Nearest/k-nearest column search over `Xt` (`p × m`, one point per column) by
explicit `@inbounds @simd` distance loops rather than a NearestNeighbors
tree. `Xt` is a plain `Matrix{Float64}`, so the search compiles once no
matter how large `p` is — unlike `KDTree`/`BruteTree`, which specialize on
`SVector{p, Float64}` and become impractical to compile above a few
thousand columns (issue #72).
"""
struct MatrixSearch
  Xt::Matrix{Float64}    # p × m, the columns handed to the structure
  d::Vector{Float64}     # scratch of length m (squared distances)
end

MatrixSearch(Xt::Matrix{Float64}) = MatrixSearch(Xt, Vector{Float64}(undef, size(Xt, 2)))

# Squared distance from column j of Xt to q, an explicit loop so it
# compiles once for every p (unlike the tree's SVector specialization).
function _sqdist(Xt::Matrix{Float64}, j::Int, q)
  acc = 0.0
  @inbounds @simd for i in axes(Xt, 1)
    diff = Xt[i, j] - q[i]
    acc += diff * diff
  end
  return acc
end

"""
    _nn(s::MatrixSearch, q, skip) -> (idx, dist)

Nearest column of `s.Xt` to `q` with `!skip(j)`, by an explicit distance
loop; allocation-free. `dist` is Euclidean, matching NearestNeighbors' `nn`.
"""
function _nn(s::MatrixSearch, q, skip)
  Xt = s.Xt
  best = 0
  bestsq = Inf
  @inbounds for j in axes(Xt, 2)
    skip(j) && continue
    sq = _sqdist(Xt, j, q)
    if sq < bestsq
      bestsq = sq
      best = j
    end
  end
  best == 0 && throw(ArgumentError("no columns available: skip excluded all of them"))
  return best, sqrt(bestsq)
end

"""
    _knn(s::MatrixSearch, q, k, skip) -> (idxs, dists)

The `k` columns of `s.Xt` nearest to `q` with `!skip(j)`, by increasing
distance; allocates only the returned index vector. `dists` are Euclidean,
matching NearestNeighbors' `knn`.
"""
function _knn(s::MatrixSearch, q, k::Int, skip)
  Xt, d = s.Xt, s.d
  navailable = 0
  @inbounds for j in axes(Xt, 2)
    if skip(j)
      d[j] = Inf
    else
      d[j] = _sqdist(Xt, j, q)
      navailable += 1
    end
  end
  k <= navailable ||
    throw(ArgumentError("k=$k exceeds the number of available columns ($navailable)"))
  idxs = partialsortperm(d, 1:k)
  return idxs, sqrt.(view(d, idxs))
end

# NearestNeighbors path: same signatures as the MatrixSearch methods above,
# so twinning and select_nearest can call either structure through _nn/_knn.
_nn(tree::NNTree, q, skip) = nn(tree, q, skip)
_knn(tree::NNTree, q, k::Int, skip) = knn(tree, q, k, true, skip)

"""
    select_nearest(data, points; search = _nearest_search(size(data, 2))) -> Vector{Int}

Indices of the data rows selected for the support points, in support-point
order. `data` and `points` have observations in rows.

`search` is `:kdtree` (a NearestNeighbors k-d tree, doubling `k` on a query
that returns only claimed rows) or `:matrix` (one [`MatrixSearch`](@ref)
pass per point); the default follows `NEAREST_BRUTE_FORCE_DIMENSION`. Both
implement the same claim-the-nearest-unclaimed-row rule and agree except on
exact ties, which continuous data does not produce.

When the optimized `points` sit closer to their own starting row than to
any other row, this returns the initial random sample unchanged. That is
expected in high dimension, where the displacement is small relative to the
row spacing (see the Benchmarks page).
"""
function select_nearest(
  data::Matrix{Float64},
  points::Matrix{Float64};
  search::Symbol = _nearest_search(size(data, 2)),
)
  size(data, 2) == size(points, 2) ||
    throw(ArgumentError("data and points must have the same number of columns."))
  size(points, 1) <= size(data, 1) ||
    throw(ArgumentError("Cannot select more points than data rows."))
  search in (:kdtree, :matrix) ||
    throw(ArgumentError("search must be :kdtree or :matrix, got :$search"))

  return search === :matrix ? _select_nearest_matrix(data, points) :
         _select_nearest_kdtree(data, points)
end

# Current k-d tree path, unchanged: query k nearest, double k when every
# returned neighbor is already claimed.
function _select_nearest_kdtree(data::Matrix{Float64}, points::Matrix{Float64})
  n = size(data, 1)
  tree = KDTree(permutedims(data))   # NearestNeighbors stores points in columns
  selected = Vector{Int}(undef, size(points, 1))
  used = falses(n)

  for j in axes(points, 1)
    query = vec(points[j, :])
    k = 1
    pick = 0
    while pick == 0
      idxs, _ = knn(tree, query, k, true)
      for idx in idxs
        if !used[idx]
          pick = idx
          break
        end
      end
      pick == 0 && (k = min(2 * k, n))
    end
    used[pick] = true
    selected[j] = pick
  end
  return selected
end

# MatrixSearch path: skip claimed rows directly, one pass per point (no
# k-doubling needed since _nn already excludes them). Both data and points
# are transposed once so every query is a contiguous column view -- a
# strided row view (`view(points, j, :)`) defeats the `@simd` loop in
# `_sqdist`, since it reads with a stride instead of unit stride.
function _select_nearest_matrix(data::Matrix{Float64}, points::Matrix{Float64})
  Xt = permutedims(data)     # p × n, columns are data rows
  Pt = permutedims(points)   # p × m, columns are query points
  search = MatrixSearch(Xt)
  used = falses(size(data, 1))
  selected = Vector{Int}(undef, size(points, 1))

  for j in axes(points, 1)
    query = view(Pt, :, j)
    idx, _ = _nn(search, query, i -> used[i])
    used[idx] = true
    selected[j] = idx
  end
  return selected
end
