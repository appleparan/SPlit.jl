"""
Sequential nearest-neighbor selection: each support point, in order, claims
its nearest not-yet-selected data row (Joseph & Vakayil 2022). Served by a
k-d tree; when every returned neighbor is already claimed, the query doubles
`k` and retries.
"""

using NearestNeighbors

"""
    select_nearest(data, points) -> Vector{Int}

Indices of the data rows selected for the support points, in support-point
order. `data` and `points` have observations in rows.

When the optimized `points` sit closer to their own starting row than to
any other row, this returns the initial random sample unchanged. That is
expected in high dimension, where the displacement is small relative to the
row spacing (see the Benchmarks page).
"""
function select_nearest(data::Matrix{Float64}, points::Matrix{Float64})
  size(data, 2) == size(points, 2) ||
    throw(ArgumentError("data and points must have the same number of columns."))
  size(points, 1) <= size(data, 1) ||
    throw(ArgumentError("Cannot select more points than data rows."))

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
