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
    include("weighted_truncation.jl")
    include("orth_columns_allocations.jl")
    include("enrichment.jl")
    include("bug.jl")
    include("augmented_threads.jl")
    include("parallel_bug.jl")
    include("parallel_blocks.jl")
    include("bug_preparation.jl")
    include("bug_workspaces.jl")
    include("grouped_contractions.jl")
    include("environment_reuse.jl")
    include("adaptive.jl")
    include("parallel_adaptive.jl")
    include("timing.jl")
end
