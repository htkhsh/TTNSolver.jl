# Pre-workspace/threading canonical assembly; benchmark oracle only.
# Exact root-orthogonal x + alpha*term*y. Contract the two input blocks into
# each new child basis separately, then QR their parent-column concatenation.
# Never build the zero-filled three-index direct-sum core. Inputs may be
# noncanonical and are read-only; all output storage is independently owned.
function reference_canonical_add(x,y,term,alpha)
    TTNSolver._check_layout(x,y)
    tree = x.tree
    coefficient = alpha*term.coefficient
    if TTNSolver._isleaf(tree,tree.root)
        U = x.leaves[tree.root] + coefficient*TTNSolver._local_action(term,1,y.leaves[tree.root])
        return TTNTensor(tree,Dict(tree.root=>U),Dict{Int,Array{ComplexF64,3}}())
    end
    modes = zeros(Int,length(tree.children))
    for (mode,node) in enumerate(tree.leaf_order)
        modes[node] = mode
    end
    leaves = Dict{Int,Matrix{ComplexF64}}()
    cores = Dict{Int,Array{ComplexF64,3}}()
    rx = Vector{Matrix{ComplexF64}}(undef,length(tree.children))
    ry = similar(rx)
    for node in tree.postorder
        leaf = TTNSolver._isleaf(tree,node)
        if leaf
            a = x.leaves[node]
            b = TTNSolver._local_action(term,modes[node],y.leaves[node])
        else
            left,right = tree.children[node]
            a = TTNSolver._project_children(x.cores[node],rx[left],rx[right])
            b = TTNSolver._project_children(y.cores[node],ry[left],ry[right])
            if node == tree.root
                cores[node] = a+coefficient*b
                continue
            end
            dl,dr = size(a,2),size(a,3)
            a,b = TTNSolver._basis_matrix(a),TTNSolver._basis_matrix(b)
        end
        M = hcat(a,b)
        F = qr!(M)
        rank = min(size(M)...)
        Q,R = Matrix(F.Q)[:,1:rank],Matrix(F.R)
        split = size(a,2)
        rx[node],ry[node] = R[:,1:split],R[:,split+1:end]
        if leaf
            leaves[node] = Q
        else
            cores[node] = reshape(Matrix(transpose(Q)),rank,dl,dr)
        end
    end
    return TTNTensor(tree,leaves,cores)
end

# The same term order, rank cap and weighted error budget as the original
# termwise block sum. Compression gets an already canonical, privately owned
# sum. Defer post-cut QR to the next exact canonical addition, then perform it
# once at the end for a stable root norm. Intermediate sums may be noncanonical.
# Do not replace this with ||b||²+||Ax||²-2real<b,Ax>: that loses small residuals.
function reference_canonical_residual(A,x,b,budget,maxrank)
    residual = copy(b)
    bound = 0.0
    perterm = budget/max(1,length(A.terms))
    for term in A.terms
        iszero(term.coefficient) && continue
        residual = reference_canonical_add(residual,x,term,-1)
        cut = TTNSolver._truncate_canonical!(residual;atol=perterm,maxrank,canonicalize=false)
        bound += cut.discarded_bound
    end
    orthogonalize!(residual)
    root = residual.tree.root
    rootdata = TTNSolver._isleaf(residual.tree,root) ? residual.leaves[root] : residual.cores[root]
    return residual,norm(rootdata)+bound,bound
end
