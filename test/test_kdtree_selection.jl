using Test
using SPlit
using Random
using LinearAlgebra

# Reference implementation: sequential nearest-neighbor with removal.
function brute_force_select(data, points)
  available = collect(1:size(data, 1))
  selected = Int[]
  for j in axes(points, 1)
    dists = [norm(view(data, i, :) .- view(points, j, :)) for i in available]
    pick = available[argmin(dists)]
    push!(selected, pick)
    filter!(!=(pick), available)
  end
  return selected
end

@testset "select_nearest" begin
  @testset "matches brute force on continuous data" begin
    for seed in (1, 2, 3)
      rng = MersenneTwister(seed)
      data = randn(rng, 300, 3)
      points = randn(rng, 40, 3) .* 0.8
      @test SPlit.select_nearest(data, points) == brute_force_select(data, points)
      @test SPlit.select_nearest(data, points; search = :matrix) ==
            brute_force_select(data, points)
      @test SPlit.select_nearest(data, points; search = :kdtree) ==
            SPlit.select_nearest(data, points; search = :matrix)
    end
  end

  @testset "no duplicates, valid range" begin
    rng = MersenneTwister(4)
    data = randn(rng, 100, 2)
    points = randn(rng, 100, 2)   # k == n forces the requery path to exhaust
    for search in (:kdtree, :matrix)
      sel = SPlit.select_nearest(data, points; search)
      @test length(sel) == 100
      @test sort(sel) == collect(1:100)
    end
  end

  @testset "the default search structure follows NEAREST_BRUTE_FORCE_DIMENSION" begin
    p = SPlit.NEAREST_BRUTE_FORCE_DIMENSION
    rng = MersenneTwister(21)
    data = randn(rng, 200, p)
    points = randn(rng, 30, p)
    @test SPlit.select_nearest(data, points) ==
          SPlit.select_nearest(data, points; search = :matrix)
    datalow = randn(rng, 200, p - 1)
    pointslow = randn(rng, 30, p - 1)
    @test SPlit.select_nearest(datalow, pointslow) ==
          SPlit.select_nearest(datalow, pointslow; search = :kdtree)
  end

  @testset "argument validation" begin
    @test_throws ArgumentError SPlit.select_nearest(randn(5, 2), randn(3, 3))
    @test_throws ArgumentError SPlit.select_nearest(randn(3, 2), randn(5, 2))
    @test_throws ArgumentError SPlit.select_nearest(
      randn(5, 2),
      randn(3, 2);
      search = :bogus,
    )
  end
end
