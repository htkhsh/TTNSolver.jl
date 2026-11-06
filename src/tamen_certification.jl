# Exact root-orthogonal x + alpha*term*y. Contract the two input blocks into
# each new child basis separately, then QR their parent-column concatenation.
# Never build the zero-filled three-index direct-sum core. Inputs may be
# noncanonical and are read-only; all output storage is independently owned.
function _tamen_add_action_canonical(x,y,term,alpha;workspace=nothing,threaded=false)
    return _with_residual_workspace(workspace) do work
        _residual_add!(work,x,y,term,alpha,threaded)
    end
end

function _residual_add!(work,x,y,term,alpha,threaded)
    _check_layout(x,y)
    tree = x.tree
    coefficient = alpha*term.coefficient
    n=length(tree.children)
    _prepare_residual_nodes!(work,n)
    modes = zeros(Int,n)
    for (mode,node) in enumerate(tree.leaf_order)
        modes[node] = mode
    end
    leaves = Vector{Matrix{ComplexF64}}(undef,n)
    cores = Vector{Array{ComplexF64,3}}(undef,n)
    rx = Vector{AbstractMatrix{ComplexF64}}(undef,n)
    ry = similar(rx)
    ranks=zeros(Int,n)
    cost=zeros(Float64,n)
    for node in tree.postorder
        columns=_isleaf(tree,node) ? size(x.leaves[node],2)+size(y.leaves[node],2) :
                                    size(x.cores[node],1)+size(y.cores[node],1)
        if _isleaf(tree,node)
            rows=size(x.leaves[node],1)
        else
            left,right=tree.children[node]
            rows=ranks[left]*ranks[right]
            cost[node]=cost[left]+cost[right]
        end
        ranks[node]=min(rows,columns)
        cost[node]+=Float64(rows)*columns*ranks[node]
    end
    function build!(node)
        leaf=_isleaf(tree,node)
        scratch=work.nodes[node]
        if leaf
            a,b=x.leaves[node],y.leaves[node]
            px,py=size(a,2),size(b,2)
            M=_residual_buffer!(scratch,5,size(a,1),px+py)
            copyto!(view(M,:,1:px),a)
            factor=get(term.factors,modes[node],nothing)
            if isnothing(factor)
                copyto!(view(M,:,px+1:px+py),b)
            else
                mul!(view(M,:,px+1:px+py),factor,b)
            end
            if node==tree.root
                leaves[node]=reshape([M[i,1]+coefficient*M[i,2] for i in axes(M,1)],:,1)
                return
            end
        else
            left,right = tree.children[node]
            dl,dr=size(rx[left],1),size(rx[right],1)
            px,py=size(x.cores[node],1),size(y.cores[node],1)
            M=_residual_buffer!(scratch,5,dl*dr,px+py)
            _residual_project_block!(view(M,:,1:px),x.cores[node],rx[left],rx[right],scratch)
            _residual_project_block!(view(M,:,px+1:px+py),y.cores[node],ry[left],ry[right],scratch)
            if node == tree.root
                cores[node]=reshape([M[i,1]+coefficient*M[i,2] for i in axes(M,1)],1,dl,dr)
                return
            end
        end
        Q,R=_residual_qr!(scratch,M)
        rx[node],ry[node]=view(R,:,1:px),view(R,:,px+1:px+py)
        if leaf
            leaves[node] = Matrix(Q)
        else
            cores[node] = permutedims(reshape(Q,dl,dr,size(Q,2)),(3,1,2))
        end
        return nothing
    end
    function visit!(node,budget)
        if !_isleaf(tree,node)
            left,right=tree.children[node]
            lb=round(Int,budget*cost[left]/(cost[left]+cost[right]))
            if budget>1 && 0<lb<budget
                work.nodes[node].tasks+=1
                task=Threads.@spawn try visit!(left,lb); nothing catch e; e end
                right_error=try visit!(right,budget-lb); nothing catch e; e end
                left_error=fetch(task)
                isnothing(left_error) || throw(left_error)
                isnothing(right_error) || throw(right_error)
            elseif lb==0
                visit!(left,1)
                visit!(right,budget)
            else
                visit!(right,1)
                visit!(left,budget)
            end
        end
        build!(node)
    end
    try
        visit!(tree.root,threaded ? Threads.nthreads(:default) : 1)
        # Publish owned arrays only on the caller, after every task has joined.
        return TTNTensor(tree,Dict(node=>leaves[node] for node in tree.leaf_order),
            Dict(node=>cores[node] for node in tree.postorder if !_isleaf(tree,node)))
    finally
        # All visit! failure paths join before unwinding. Counters are reduced
        # here, never concurrently mutated by worker tasks.
        work.grows+=sum(node.grows for node in work.nodes)
        work.reuses+=sum(node.reuses for node in work.nodes)
        work.tasks+=sum(node.tasks for node in work.nodes)
    end
end

# The same term order, rank cap and weighted error budget as the original
# termwise block sum. Compression gets an already canonical, privately owned
# sum. Defer post-cut QR to the next exact canonical addition, then perform it
# once at the end for a stable root norm. Intermediate sums may be noncanonical.
# Do not replace this with ||b||²+||Ax||²-2real<b,Ax>: that loses small residuals.
function _tamen_residual(A,x,b,budget,maxrank;workspace=nothing,threaded=false)
    if isnothing(workspace)
        return _residual_certificate(A,x,b,budget,maxrank,nothing,threaded)
    end
    return _with_residual_workspace(workspace) do work
        _residual_certificate(A,x,b,budget,maxrank,work,threaded)
    end
end

# Experimental one-application certificate. Its error allowance includes any
# operator compression as ||A-W||_F ||x||; never accept on a compressed operator
# residual alone. The small residual is assembled explicitly to avoid squared-
# norm subtraction near convergence. Threading currently uses the sum builder.
function _tamen_residual(A::TreeTensorOperator,x,b,budget,maxrank;workspace=nothing,threaded=false)
    return _with_residual_workspace(workspace) do work
        started=_residual_tick(work.timings)
        acted=apply(A,x)
        residual=_residual_add!(work,b,acted,ProductTerm(1,Dict()),-1,threaded)
        _residual_record!(work.timings,Val(:addition_seconds),started)
        cut=_residual_truncate!(residual,work.cut;atol=budget,maxrank,timings=work.timings)
        bound=cut.discarded_bound+A.discarded_bound*norm(x)
        started=_residual_tick(work.timings)
        orthogonalize!(residual)
        root=residual.tree.root
        data=_isleaf(residual.tree,root) ? residual.leaves[root] : residual.cores[root]
        _residual_record!(work.timings,Val(:final_qr_seconds),started)
        return residual,norm(data)+bound,bound
    end
end

function _residual_certificate(A,x,b,budget,maxrank,workspace,threaded)
    residual = copy(b)
    bound = 0.0
    perterm = budget/max(1,length(A.terms))
    for term in A.terms
        iszero(term.coefficient) && continue
        work=isnothing(workspace) ? _ResidualWorkspace() : workspace
        started=_residual_tick(work.timings)
        residual = _residual_add!(work,residual,x,term,-1,threaded)
        _residual_record!(work.timings,Val(:addition_seconds),started)
        cut = _residual_truncate!(residual,work.cut;atol=perterm,maxrank,timings=work.timings)
        bound += cut.discarded_bound
    end
    timings=isnothing(workspace) ? nothing : workspace.timings
    started=_residual_tick(timings)
    orthogonalize!(residual)
    _residual_record!(timings,Val(:final_qr_seconds),started)
    root = residual.tree.root
    rootdata = _isleaf(residual.tree,root) ? residual.leaves[root] : residual.cores[root]
    return residual,norm(rootdata)+bound,bound
end
