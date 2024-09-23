using LinearAlgebra
using Random
using NearestNeighbors

# Function to print progress
function print_progress(percent::Int, num_threads::Int)
  print(
    "\rOptimizing <$num_threads threads> [$(repeat('+', percent ÷ 5))$(repeat(' ', 20 - percent ÷ 5))] $percent%",
  )
end

"""
    sp_julia(des_num::Int, dim_num::Int, ini::Matrix{Float64}, distsamp::Matrix{Float64}, 
             bd::Matrix{Float64}, point_num::Int, it_max::Int, tol::Float64, 
             num_proc::Int, n0::Float64, wts::Vector{Float64}, rnd_flg::Bool; tree_type=:KDTree)

Optimizes a space-filling design using an iterative algorithm. The optimization process uses 
either a KDTree or BallTree for nearest neighbor searches to ensure the design points are well distributed.

# Arguments
- `des_num::Int`: Number of design points to be optimized.
- `dim_num::Int`: Number of dimensions for each design point.
- `ini::Matrix{Float64}`: Initial design points matrix of size (`des_num`, `dim_num`).
- `distsamp::Matrix{Float64}`: A matrix of sampled points for distance calculation.
- `bd::Matrix{Float64}`: Boundary conditions for each dimension in the form of a matrix with two columns, where each row corresponds to a dimension.
- `point_num::Int`: Number of random points to be used in each iteration.
- `it_max::Int`: Maximum number of iterations for the optimization process.
- `tol::Float64`: Tolerance level to determine convergence.
- `num_proc::Int`: Number of parallel threads to use for computation.
- `n0::Float64`: A parameter controlling the influence of current iteration in the optimization process.
- `wts::Vector{Float64}`: Weights for each sampled point.
- `rnd_flg::Bool`: If true, enables random sampling of points.
- `tree_type::Symbol`: The type of tree to use for nearest neighbor search; either `:KDTree` or `:BallTree`.

# Returns
- `Matrix{Float64}`: A matrix of optimized design points of size (`des_num`, `dim_num`).
"""
function sp_julia(
  des_num::Int,
  dim_num::Int,
  ini::Matrix{Float64},
  distsamp::Matrix{Float64},
  bd::Matrix{Float64},
  point_num::Int,
  it_max::Int,
  tol::Float64,
  num_proc::Int,
  n0::Float64,
  wts::Vector{Float64},
  rnd_flg::Bool;
  tree_type::Symbol = :KDTree,
)

  Threads.nthreads() == num_proc || Threads.@threads :num_proc

  # Initialization
  it_num = 0
  des = copy(ini)
  nug = 0.0
  rng = MersenneTwister()
  percent_complete = 0

  # Iterative Optimization
  while it_num < it_max
    percent = Int(100 * (it_num + 1) / it_max)
    if percent > percent_complete
      print_progress(percent, num_proc)
      percent_complete = percent
    end

    # Random sampling
    rnd_indices = rnd_flg ? rand(rng, 1:size(distsamp, 1), point_num) : 1:point_num
    rnd = distsamp[rnd_indices, :]
    rnd_wts = wts[rnd_indices]

    # Create KDTree or BallTree
    tree = tree_type == :KDTree ? KDTree(rnd) : BallTree(rnd)

    # Parallel Update
    des_up = copy(des)
    Threads.@threads for m = 1:des_num
      xprime = zeros(dim_num)

      # Contributions from other design points
      for o = 1:des_num
        if o != m
          tmpvec = des[m, :] - des[o, :]
          tmptol = norm(tmpvec) + nug * eps()
          xprime .+= tmpvec ./ tmptol
        end
      end

      xprime .*= point_num / des_num

      # Nearest neighbor interactions
      indices, distances = inrange(tree, des[m, :], tol)

      curconst = 0.0
      for i in eachindex(indices)
        tmptol = distances[i] + nug * eps()
        curconst += rnd_wts[i] / tmptol
        xprime .+= rnd_wts[i] .* rnd[indices[i], :] ./ tmptol
      end

      denom = (1.0 - (n0 / (it_num + n0))) + (n0 / (it_num + n0)) * curconst
      xprime .=
        ((1.0 - (n0 / (it_num + n0))) .* des[m, :] .+ (n0 / (it_num + n0)) .* xprime) ./
        denom

      # Enforce boundary conditions
      xprime .= clamp.(xprime, bd[:, 1], bd[:, 2])

      # Update the design point
      des_up[m, :] = xprime
    end

    # Check convergence
    maxdiff = maximum(norm.(des_up .- des, dims = 2))
    des .= des_up

    if maxdiff < tol
      println("\nTolerance level reached.")
      break
    end

    it_num += 1
  end

  println()
  return des
end
