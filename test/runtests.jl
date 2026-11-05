using Test
using Random
using LinearAlgebra
using TTNSolver

@testset "TTNSolver" begin
    include("trees.jl")
    include("states.jl")
    include("algebra.jl")
    include("operators.jl")
    include("local_contractions.jl")
    include("truncation.jl")
    include("enrichment.jl")
    include("bug.jl")
    include("bug_preparation.jl")
    include("bug_workspaces.jl")
    include("adaptive.jl")
    include("timing.jl")
end
