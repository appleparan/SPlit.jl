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
@testset "TwinningSplitter" begin
  @testset "construction, validation, show" begin
    s = TwinningSplitter()
    @test s isa AbstractSplitter
    @test s.kernel isa EnergyKernel && s.ratio == 0.2 && s.start === :farthest
    @test TwinningSplitter(ratio = 1 // 4, start = 7).start == 7
    @test_throws ArgumentError TwinningSplitter(ratio = 0.0)
    @test_throws ArgumentError TwinningSplitter(ratio = 1.0)
    @test_throws ArgumentError TwinningSplitter(start = :middle)
    @test_throws ArgumentError TwinningSplitter(start = 0)
    @test occursin("TwinningSplitter(ratio=0.2, start=:farthest)", sprint(show, s))
  end

  @testset "datasplit: partition, sizes, honest report" begin
    data = randn(MersenneTwister(10), 300, 3)
    r = datasplit(TwinningSplitter(), data)
    @test length(test_indices(r)) == 60 && length(train_indices(r)) == 240
    @test sort(vcat(train_indices(r), test_indices(r))) == 1:300
    @test r.converged && r.iterations == 60 && r.selected === :test
    @test r.method isa TwinningSplitter && r.method.kernel isa EnergyKernel
    @test datasplit(TwinningSplitter(), data).test_indices == r.test_indices   # deterministic, no rng
  end

  @testset "ratio > 0.5 puts the selected rows in train" begin
    data = randn(MersenneTwister(11), 200, 2)
    r = datasplit(TwinningSplitter(ratio = 0.7), data)
    @test length(test_indices(r)) == 140 && r.selected === :train
    @test train_indices(r) == selectrows(TwinningSplitter(ratio = 0.7), data, 60)
  end

  @testset "selectrows is the selected side, in formation order, starting at start" begin
    data = randn(MersenneTwister(12), 150, 2)
    idx = selectrows(TwinningSplitter(start = 42), data, 30)
    @test length(idx) == 30 && allunique(idx) && idx[1] == 42
    @test sort(idx) == sort(test_indices(datasplit(TwinningSplitter(start = 42), data)))
    X = SPlit.preprocess(data)
    @test idx == first.(SPlit._twin_groups(X, 30, 42, MersenneTwister(0)))
    @test_throws ArgumentError selectrows(TwinningSplitter(start = 151), data, 30)
    # n > N/2 is allowed (single-row groups)
    @test length(selectrows(TwinningSplitter(), data, 120)) == 120
  end

  @testset ":random start is reproducible under the same rng and differs across seeds" begin
    data = randn(MersenneTwister(13), 200, 2)
    a = selectrows(TwinningSplitter(start = :random, rng = MersenneTwister(1)), data, 40)
    b = selectrows(TwinningSplitter(start = :random, rng = MersenneTwister(1)), data, 40)
    c = selectrows(TwinningSplitter(start = :random, rng = MersenneTwister(2)), data, 40)
    @test a == b
    @test a[1] != c[1] || a != c
  end

  @testset "DataFrame with categoricals, vector input, duplicate rows" begin
    df = DataFrame(x = randn(MersenneTwister(14), 90), g = repeat(["a", "b", "c"], 30))
    r = datasplit(TwinningSplitter(), df)
    @test length(test_indices(r)) == 18
    v = randn(MersenneTwister(15), 50)
    @test length(test_indices(datasplit(TwinningSplitter(), v))) == 10
    dup = vcat(randn(MersenneTwister(16), 20, 2), randn(MersenneTwister(16), 20, 2))
    rd = datasplit(TwinningSplitter(), dup)
    @test sort(vcat(train_indices(rd), test_indices(rd))) == 1:40
  end

  @testset "weights and reference are rejected" begin
    data = randn(MersenneTwister(17), 100, 2)
    @test_throws ArgumentError datasplit(TwinningSplitter(), data; weights = rand(100))
    @test_throws ArgumentError selectrows(
      TwinningSplitter(),
      data,
      20;
      reference = data[1:50, :],
    )
    @test_throws ArgumentError datasplit(TwinningSplitter(), data; weights = ones(100))
  end

  @testset "compare and splitquality accept a TwinningSplitter" begin
    data = randn(MersenneTwister(18), 200, 2)
    c = compare([TwinningSplitter(), HerdingSplitter(kernel = EnergyKernel())], data)
    df = DataFrame(c)
    @test df.method == ["TwinningSplitter", "HerdingSplitter"]
    @test df.kernel == ["EnergyKernel", "EnergyKernel"]
    @test all(isfinite, c.qualities)
    r = datasplit(TwinningSplitter(), data)
    @test splitquality(data, r) ≈ c.qualities[1]
  end
end
