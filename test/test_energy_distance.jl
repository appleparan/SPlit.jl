
struct FooDist <: PreMetric end # Julia 1.0 Compat: struct definition must be put in global scope

@testset "result_type" begin
    foodist(a, b) = a + b
    (::FooDist)(a, b) = foodist(a, b)
    for (Ta, Tb) in [
        (Int, Int),
        (Int, Float64),
        (Float32, Float32),
        (Float32, Float64),
    ]
        A, B = rand(Ta, 2, 3), rand(Tb, 2, 3)
        @test result_type(FooDist(), A, B) == result_type(FooDist(), Ta, Tb)
        @test result_type(foodist, A, B) == result_type(foodist, Ta, Tb) == typeof(foodist(oneunit(Ta), oneunit(Tb)))

        a, b = rand(Ta), rand(Tb)
        @test result_type(FooDist(), a, b) == result_type(FooDist(), Ta, Tb)
        @test result_type(foodist, a, b) == result_type(foodist, Ta, Tb) == typeof(foodist(oneunit(Ta), oneunit(Tb)))
    end
end

@testset "Wasserstein (Earth mover's) distance" begin
    Random.seed!(123)
    for T in [Float64]
    # for T in [Float32, Float64]
        N = 5
        u = rand(T, N)
        v = rand(T, N)
        u_weights = rand(T, N)
        v_weights = rand(T, N)

        dist = Wasserstein(u_weights, v_weights)

        test_pairwise(dist, u, v, T)

        @test evaluate(dist, u, v) === wasserstein(u, v, u_weights, v_weights)
        @test dist(u, v) === wasserstein(u, v, u_weights, v_weights)

        @test_throws ArgumentError wasserstein([], [])
        @test_throws ArgumentError wasserstein([], v)
        @test_throws ArgumentError wasserstein(u, [])
        @test_throws DimensionMismatch wasserstein(u, v, u_weights[1:end-1], v_weights)
        @test_throws DimensionMismatch wasserstein(u, v, u_weights, v_weights[1:end-1])
        @test_throws ArgumentError wasserstein(u, v, -u_weights, v_weights)
        @test_throws ArgumentError wasserstein(u, v, u_weights, -v_weights)

        # # TODO: Needs better/more correctness tests
        # @test wasserstein(u, v)                       ≈ 0.2826796049559892
        # @test wasserstein(u, v, u_weights, v_weights) ≈ 0.28429147575475444
    end

end