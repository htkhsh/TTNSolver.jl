using TTNSolver
using LinearAlgebra

# These physical leaves can later represent HEOM, TFD or other tensor factors.
tree = balanced_tree(3)
state = product_state(tree, [ComplexF64[2im, 3], [1, 0, 2], [1, -im]])
reference = dense(state)
initial_norm = norm(state)
orthogonalize!(state)

@assert isapprox(dense(state), reference; atol=1e-12, rtol=1e-12)
@assert isapprox(norm(state), initial_norm; atol=1e-12, rtol=1e-12)
println("Physical dimensions: ", leafdims(state))
println("Node ranks: ", tree_ranks(state))
println("Norm before / after QR: ", initial_norm, " / ", norm(state))
println("Dense reconstruction error: ", norm(dense(state) - reference))
