using Test
using SPlit
using Random
using Statistics
using DataFrames
using LinearAlgebra

# Algorithm 1 of Vakayil & Joseph (2022) with brute-force distances and
# explicit masking, under the group-size rule of the spec. Ties go to the
# lowest row index (irrelevant on continuous data).
function naive_twin_groups(X::Matrix{Float64}, n::Int, start::Int)
  N = size(X, 1)
  sizes = SPlit._group_sizes(N, n)
  alive = trues(N)
  d(a, b) = norm(view(X, a, :) .- view(X, b, :))
  groups = Vector{Vector{Int}}()
  u = start
  far = start
  for i = 1:n
    if i > 1
      cands = findall(alive)
      u = cands[argmin([d(far, j) for j in cands])]
    end
    others = [j for j in findall(alive) if j != u]
    sort!(others; by = j -> d(u, j))
    g = vcat(u, others[1:(sizes[i]-1)])
    alive[g] .= false
    far = g[end]
    push!(groups, g)
  end
  return groups
end

@testset "twinning core" begin
  @testset "group sizes: r or r + 1, sum to N, all r when N = r·n" begin
    @test SPlit._group_sizes(50, 10) == fill(5, 10)
    s = SPlit._group_sizes(137, 23)
    @test length(s) == 23 && sum(s) == 137
    @test all(x -> x in (5, 6), s) && count(==(6), s) == 137 - 5 * 23
    @test sum(SPlit._group_sizes(11, 2)) == 11 && SPlit._group_sizes(11, 2) == [5, 6]
    @test SPlit._group_sizes(7, 7) == fill(1, 7)
    # oversized groups are spread along the chain, not concentrated at the end
    s2 = SPlit._group_sizes(100, 40)   # r = 2, extra = 20
    @test s2[1:2] == [2, 3] || s2[1:2] == [3, 2]
  end

  @testset "start row" begin
    X = [0.0 0.0; 3.0 4.0; -1.0 0.0]
    @test SPlit._start_row(:farthest, X, MersenneTwister(1)) == 2
    @test SPlit._start_row(:farthest, [1.0 0.0; 0.0 1.0], MersenneTwister(1)) == 1  # tie → lowest
    @test SPlit._start_row(3, X, MersenneTwister(1)) == 3
    r1 = SPlit._start_row(:random, X, MersenneTwister(5))
    @test r1 == SPlit._start_row(:random, X, MersenneTwister(5)) && r1 in 1:3
    @test_throws ArgumentError SPlit._start_row(4, X, MersenneTwister(1))
    @test_throws ArgumentError SPlit._start_row(0, X, MersenneTwister(1))
    @test_throws ArgumentError SPlit._start_row(:middle, X, MersenneTwister(1))
  end

  @testset "groups partition the rows and follow the size rule" begin
    X = SPlit.preprocess(randn(MersenneTwister(1), 137, 3))
    groups = SPlit._twin_groups(X, 23, :farthest, MersenneTwister(0))
    @test length(groups) == 23
    @test sort(reduce(vcat, groups)) == 1:137
    @test length.(groups) == SPlit._group_sizes(137, 23)
    @test first(groups[1]) == argmax(vec(sum(abs2, X; dims = 2)))
  end

  @testset "kd-tree, masking, and rebuilds implement the definition" begin
    for (N, n, p, seed) in
        ((137, 23, 3, 2), (60, 30, 2, 3), (100, 20, 4, 4), (90, 60, 2, 5))
      X = SPlit.preprocess(randn(MersenneTwister(seed), N, p))
      expected = naive_twin_groups(X, n, 1)
      @test SPlit._twin_groups(X, n, 1, MersenneTwister(0)) == expected
      @test SPlit._twin_groups(X, n, 1, MersenneTwister(0); brute_force = true) == expected
    end
  end

  @testset "each group is u followed by its neighbors in increasing distance" begin
    X = SPlit.preprocess(randn(MersenneTwister(6), 80, 2))
    for g in SPlit._twin_groups(X, 16, :farthest, MersenneTwister(0))
      dists = [norm(X[g[1], :] .- X[j, :]) for j in g[2:end]]
      @test issorted(dists)
    end
  end

  @testset "n = N gives singleton groups, n = 1 one group" begin
    X = SPlit.preprocess(randn(MersenneTwister(7), 12, 2))
    @test length.(SPlit._twin_groups(X, 12, 1, MersenneTwister(0))) == fill(1, 12)
    @test SPlit._twin_groups(X, 1, 1, MersenneTwister(0)) ==
          [vcat(1, sort(2:12; by = j -> norm(X[1, :] .- X[j, :])))]
    @test_throws ArgumentError SPlit._twin_groups(X, 13, 1, MersenneTwister(0))
    @test_throws ArgumentError SPlit._twin_groups(X, 0, 1, MersenneTwister(0))
  end
end
