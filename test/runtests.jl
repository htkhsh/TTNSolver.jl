using Test
using Random
using LinearAlgebra
using TTNSolver

@testset "TTNSolver" begin
    include("trees.jl")
    include("states.jl")
    include("algebra.jl")
end
