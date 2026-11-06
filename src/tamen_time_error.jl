# Integrated defect on a finer temporal grid in the final spatial space.
# No second spatial ALS solve, full tensor, or global generator is needed.
function _tamen_time_error(solution,generator;threaded=false,local_operator=:channels)
    x = solution.state
    spatial_root = solution.initial.tree.root
    nt = size(x.leaves[solution.time_leaf],1)
    op = SumOfProducts([generator.dimensions;nt],generator.terms)
    initial_lift = _time_lift(solution.initial,ones(nt))
    A = _local_operator(op,x.tree,local_operator)
    up,cross = _tamen_upward(A,x,initial_lift;threaded)
    C = x.cores[x.tree.root]
    rank = size(C,2)
    L = _projected_spatial_generator(A,up,spatial_root,x.tree.root,rank)
    u0 = vec(cross[spatial_root])
    projection_error = _tamen_projection_error(solution,u0)
    frame = reshape(C,rank,size(C,3))*transpose(x.leaves[solution.time_leaf])
    coarse = solution.time_scheme == :cheb ? hcat(u0,frame) : hcat(u0,frame[:,2:end])
    fine_nt = solution.time_scheme == :cheb ? 2nt : 2nt-1
    grid = _tamen_grid(fine_nt,last(solution.nodes),solution.time_scheme)
    fine_nodes = solution.time_scheme == :cheb ? grid.nodes[2:end] : grid.nodes
    P = reduce(vcat,[transpose(_time_interpolation(solution.nodes,t,solution.time_scheme)) for t in fine_nodes])
    fine = coarse*transpose(P)
    integrated = fine .- u0 .- (L*fine*transpose(grid.M))/transpose(grid.D)
    absolute = norm(integrated)/sqrt(length(fine_nodes)) + projection_error
    scale = max(norm(solution.initial),norm(fine)/sqrt(length(fine_nodes)))
    relative = iszero(scale) ? (iszero(absolute) ? 0.0 : Inf) : absolute/scale
    return (;absolute,relative,scale,initial_projection_error=projection_error,
            fine_points=length(grid.nodes))
end

# C0' * (A ⊗ B) * C0 for subtree vectors folded by core C0.
_core_gram(C0,A,B) = _basis_matrix(C0)'*_basis_matrix(_project_children(C0,A,B))

function _isometric_basis(M;tol=1e-10)
    return norm(M'*M-I) <= tol*max(1,size(M,2))
end

"""
    _tamen_projection_error(solution,u0)

`norm(x0 - P*x0)` for the initial state `x0` and the orthogonal projector `P`
onto the final spatial basis. Every subtree vector of `x0` is split into its
projection onto the solution basis and an orthogonal remainder. Only Gram
matrices of mutually orthogonal parts are added, so no squared norms are
subtracted and small errors keep their relative accuracy. Requires
orthonormal spatial subtree bases, which tAMEn produces; otherwise this falls
back to the exact block-sum `ttn_distance` with root coefficients `u0`.
"""
function _tamen_projection_error(solution,u0)
    x,x0 = solution.state,solution.initial
    tree = x0.tree
    n = length(tree.children)
    M = Vector{Matrix{ComplexF64}}(undef,n)
    S = Vector{Matrix{ComplexF64}}(undef,n)
    G = Vector{Matrix{ComplexF64}}(undef,n)
    for node in tree.postorder
        if _isleaf(tree,node)
            V,W = x.leaves[node],x0.leaves[node]
            _isometric_basis(V) || return _projection_error_fallback(solution,u0)
            M[node] = V'*W
            E = W-V*M[node]
            S[node] = W'*W
            G[node] = E'*E
        else
            a,b = tree.children[node]
            B = _basis_matrix(x.cores[node])
            _isometric_basis(B) || return _projection_error_fallback(solution,u0)
            C0 = x0.cores[node]
            Y = _basis_matrix(_project_children(C0,M[a],M[b]))
            M[node] = B'*Y
            D = Y-B*M[node]
            S[node] = _core_gram(C0,S[a],S[b])
            # Components in span(Va⊗Vb) outside V, outside Va, and in Va but outside Vb.
            G[node] = D'*D + _core_gram(C0,G[a],S[b]) + _core_gram(C0,M[a]'*M[a],G[b])
        end
    end
    g = real(only(G[tree.root]))
    return sqrt(max(g,0.0))
end

_projection_error_fallback(solution,u0) =
    ttn_distance(solution.initial,_spatial_coefficients(solution,u0))
