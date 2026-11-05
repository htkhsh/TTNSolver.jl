module TTNSolver

using LinearAlgebra
import KrylovKit

export BinaryTree, balanced_tree
export TTNTensor, product_state, dense, leafdims, tree_ranks
export orthogonalize!, ttn_distance, enrich_basis
export ProductTerm, SumOfProducts, apply
export OperatorEnvironments, operator_environments, local_apply
export truncate!
export augmented_bug_step, LocalSolveFailure
export adaptive_evolve, AdaptiveIntegrationError

include("trees.jl")
include("states.jl")
include("algebra.jl")
include("operators.jl")
include("contractions.jl")
include("truncation.jl")
include("enrichment.jl")
include("bug_workspaces.jl")
include("bug.jl")
include("timing.jl")
include("adaptive.jl")

end
