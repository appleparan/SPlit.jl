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
    end
  end

  @testset "no duplicates, valid range" begin
    rng = MersenneTwister(4)
    data = randn(rng, 100, 2)
    points = randn(rng, 100, 2)   # k == n forces the requery path to exhaust
    sel = SPlit.select_nearest(data, points)
    @test length(sel) == 100
    @test sort(sel) == collect(1:100)
  end

  @testset "argument validation" begin
    @test_throws ArgumentError SPlit.select_nearest(randn(5, 2), randn(3, 3))
    @test_throws ArgumentError SPlit.select_nearest(randn(3, 2), randn(5, 2))
  end
end
