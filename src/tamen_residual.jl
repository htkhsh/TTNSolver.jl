# The auxiliary TTN approximates b-Ax; it never certifies its norm. All
# environments below have a residual bra and a fixed solution/RHS ket. They
# belong to one call, and are rebuilt when a residual basis changes.
function _tamen_residual_seed(x,rank)
    rng = Random.MersenneTwister(73019)
    tree = x.tree
    leaves = Dict{Int,Matrix{ComplexF64}}()
    cores = Dict{Int,Array{ComplexF64,3}}()
    ranks = zeros(Int,length(tree.children))
    for node in tree.postorder
        if _isleaf(tree,node)
            n = size(x.leaves[node],1)
            r = node == tree.root ? 1 : min(rank,n)
            leaves[node] = Matrix(qr(Random.randn(rng,ComplexF64,n,r)).Q)[:,1:r]
        else
            left,right = tree.children[node]
            a,b = ranks[left],ranks[right]
            r = node == tree.root ? 1 : min(rank,a*b)
            Q = Matrix(qr(Random.randn(rng,ComplexF64,a*b,r)).Q)[:,1:r]
            cores[node] = reshape(Matrix(transpose(Q)),r,a,b)
        end
        ranks[node] = r
    end
    return TTNTensor(tree,leaves,cores)
end

function _tamen_residual_sweep!(z,A,x,b)
    tree = z.tree
    modes = Dict(node=>mode for (mode,node) in enumerate(tree.leaf_order))
    up = [Matrix{ComplexF64}[] for _ in tree.children]
    cross = Vector{Matrix{ComplexF64}}(undef,length(tree.children))
    function update_up!(node)
        if _isleaf(tree,node)
            up[node] = _leaf_operators(A,modes[node],z.leaves[node],x.leaves[node])
            cross[node] = z.leaves[node]'*b.leaves[node]
        else
            left,right = tree.children[node]
            Q = _basis_matrix(z.cores[node])
            up[node] = _mixed_subtree_operators(Q,x.cores[node],up[left],up[right])
            cross[node] = Q'*_basis_matrix(_project_children(b.cores[node],cross[left],cross[right]))
        end
    end
    for node in tree.postorder
        update_up!(node)
    end
    down = [Matrix{ComplexF64}[] for _ in tree.children]
    rhsdown = Vector{Matrix{ComplexF64}}(undef,length(tree.children))
    down[tree.root] = [ones(ComplexF64,1,1) for _ in A.terms]
    rhsdown[tree.root] = ones(ComplexF64,1,1)
    local_updates = 0
    function project!(node)
        if _isleaf(tree,node)
            rhs = b.leaves[node]*transpose(rhsdown[node])
            for (k,term) in enumerate(A.terms)
                iszero(term.coefficient) && continue
                rhs .-= term.coefficient .* (_local_action(term,modes[node],x.leaves[node])*transpose(down[node][k]))
            end
            z.leaves[node] = rhs
        else
            left,right = tree.children[node]
            rhs = _parent_product(_project_children(b.cores[node],cross[left],cross[right]),rhsdown[node])
            for term in _group_core_terms(A,down[node],up[left],up[right])
                B,L,R = term.factors
                rhs .-= term.coefficient .* _parent_product(_project_children(x.cores[node],L,R),B)
            end
            z.cores[node] = rhs
        end
        local_updates += 1
    end
    function visit!(node)
        project!(node)
        _isleaf(tree,node) && return
        left,right = tree.children[node]
        for (child,sibling,mode) in ((left,right,2),(right,left,3))
            Q = _tamen_down!(z,node,child,mode)
            # Rectangular complement: z's orthonormal frame against fixed x.
            source = permutedims(x.cores[node],mode == 2 ? (2,1,3) : (3,1,2))
            down[child] = _mixed_subtree_operators(Q,source,down[node],up[sibling])
            B = _parent_product(_child_product(b.cores[node],cross[sibling],5-mode),rhsdown[node])
            rhsdown[child] = Q'*_complement_matrix(B,mode)
            visit!(child)
            _tamen_up!(z,node,child,mode)
            update_up!(child)
        end
        project!(node)
    end
    visit!(tree.root)
    return local_updates
end

# Fixed-rank ALS for I*z = b-A*x. Orthonormal insertion maps make each
# least-squares microproblem a projected RHS, so no local Krylov solve is needed.
# A private deterministic seed avoids the invisible-coupling trap of seeding
# only with x's bases and does not consume the caller's RNG state. It is a
# heuristic approximation; a global check remains mandatory for acceptance.
function _tamen_local_residual(A,x,b,previous=nothing;rank=3,sweeps=1)
    _check_operator(A,x)
    _check_layout(x,b)
    rank isa Integer && rank>=1 || throw(ArgumentError("residual rank must be positive"))
    sweeps isa Integer && sweeps>=1 || throw(ArgumentError("residual sweeps must be positive"))
    z = if isnothing(previous)
        _tamen_residual_seed(x,rank)
    else
        _check_layout(x,previous)
        candidate = orthogonalize!(copy(previous))
        maximum(tree_ranks(candidate))>rank && truncate!(candidate;maxrank=rank,atol=0.0)
        candidate
    end
    local_updates = 0
    for _ in 1:sweeps
        local_updates += _tamen_residual_sweep!(z,A,x,b)
    end
    estimate = norm(_isleaf(z.tree,z.tree.root) ? z.leaves[z.tree.root] : z.cores[z.tree.root])
    return z,(;estimate,local_updates)
end
