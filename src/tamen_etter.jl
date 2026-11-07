# Etter (2016), Definition 33: capture each child's orthonormal complement,
# then restore singular values in the virtual parent before preparing its
# sibling. Only Vt rotates cached upward messages; the actual child receives
# S*Vt. These are deliberately different frames. No inverse singular values.
# rhs has its parent index projected, but its two source-child indices open.
function _etter_child_frames(C,boundary,left_up,right_up,rhs;
                             cross_left=Matrix{ComplexF64}(I,size(C,2),size(C,2)),
                             cross_right=Matrix{ComplexF64}(I,size(C,3),size(C,3)),
                             active=(true,true),operator=nothing,node=0,threaded=false)
    orthonormal = !isnothing(operator) && _assume_orthonormal(operator)
    virtual = C
    upward = [left_up,right_up]
    cross = [cross_left,cross_right]
    frames = Any[nothing,nothing]
    for index in 1:2
        active[index] || continue
        mode,sibling = index+1,3-index
        F = svd(_complement_matrix(virtual,mode))
        Q,Vt = F.U,F.Vt
        B = _child_product(rhs,cross[sibling],5-mode)
        complement = isnothing(operator) ? _complement_operators(Q,boundary,upward[sibling];threaded) :
            _env_down(operator,node,mode,Q,boundary,upward[sibling];orthonormal,threaded)
        frames[index] = (operator=complement,
                         rhs=Q'*_complement_matrix(B,mode),
                         transfer=F.S.*Vt)
        # The virtual sibling basis is orthonormal. Keep S in the virtual
        # parent, not in its cached upward operator or cross contraction.
        if index == 1 && active[2]
            virtual = _child_product(virtual,conj(Vt),mode)
            # A unitary rotation keeps an exact identity environment exact.
            square = size(Vt,1) == size(Vt,2)
            upward[index] = [square && isone(U) ? U : conj(Vt)*U*transpose(Vt) for U in upward[index]]
            cross[index] = conj(Vt)*cross[index]
        end
    end
    return (left=frames[1],right=frames[2])
end

# Each worker owns all slots in its subtree. No TTNTensor or workspace Dict is
# mutated until every task has joined successfully. Scratch slots can change
# shape twice at a node (before/after its child solves).
function _tamen_etter_sweep!(x,A,b,time_leaf;tol,krylovdim,krylov_maxiter,workspace,threaded=false)
    tree = x.tree
    n = length(tree.children)
    modes = zeros(Int,n)
    for (mode,node) in enumerate(tree.leaf_order)
        modes[node] = mode
    end
    leaves = Vector{Matrix{ComplexF64}}(undef,n)
    cores = Vector{Array{ComplexF64,3}}(undef,n)
    for (node,U) in x.leaves
        leaves[node] = copy(U)
    end
    for (node,C) in x.cores
        cores[node] = copy(C)
    end
    environment_start = time_ns()
    up,cross = _tamen_upward(A,x,b;threaded)
    environment_seconds = (time_ns()-environment_start)/1e9
    down = [Matrix{ComplexF64}[] for _ in 1:n]
    rhsdown = Vector{Matrix{ComplexF64}}(undef,n)
    overlap = Vector{Matrix{ComplexF64}}(undef,n)
    down[tree.root] = _root_boundary(A)
    rhsdown[tree.root] = ones(ComplexF64,1,1)
    orthonormal = _assume_orthonormal(A)
    # Private copies of the slot lists; the workspace is updated after joining.
    slots = Any[(s = get(workspace.buffers,node,nothing); isnothing(s) ? nothing : copy(s)) for node in 1:n]
    allocations,reuses,solves,matvecs,iterations,spawned = (zeros(Int,n) for _ in 1:6)
    residuals = zeros(n)
    sizes = ones(Int,n)
    for node in tree.postorder
        _isleaf(tree,node) || (sizes[node] += sum(sizes[c] for c in tree.children[node]))
    end
    function buffers!(node,initial)
        cached = _slot_buffers!(slots[node],initial)
        if !isnothing(cached)
            reuses[node] += 1
            return cached
        end
        buffers = _new_tamen_buffers(initial)
        slots[node] = _slot_store(slots[node],initial,buffers)
        allocations[node] += 1
        return buffers
    end
    function solve!(node)
        leaf = _isleaf(tree,node)
        initial = leaf ? leaves[node] : cores[node]
        buffers = buffers!(node,initial)
        if leaf
            rhs = b.leaves[node]*transpose(rhsdown[node])
            action = _leaf_action(A,node,modes[node],down[node],initial;buffers)
        else
            left,right = tree.children[node]
            rhs = _parent_product(_project_children(b.cores[node],cross[left],cross[right]),rhsdown[node])
            action = _core_action(A,node,down[node],up[left],up[right],initial;buffers,threaded)
        end
        alg = KrylovKit.GMRES(;tol,krylovdim=min(krylovdim,length(initial)),maxiter=krylov_maxiter)
        result,report = KrylovKit.linsolve(action,rhs,initial,alg)
        solves[node] += 1
        matvecs[node] += report.numops
        iterations[node] += report.numiter
        # KrylovKit's GMRES recomputes r=b-A*x explicitly before reporting
        # convergence, so normres is the actual local residual of `result`.
        error = report.normres
        residuals[node] = max(residuals[node],error)
        if report.converged != 1 || !isfinite(error) || !all(isfinite,result)
            throw(TAMEnConvergenceError(:local_gmres,(;node,residual=error,
                local_solves=solves[node],matvecs=matvecs[node],iterations=iterations[node])))
        end
        if leaf
            leaves[node] = result
        else
            cores[node] = result
        end
        return nothing
    end
    function ascend!(node)
        leaf = _isleaf(tree,node)
        M = leaf ? leaves[node] : _basis_matrix(cores[node])
        rank = min(size(M)...)
        Q = Matrix(qr(M).Q)[:,1:rank]
        if leaf
            leaves[node] = Q
            up[node] = _env_leaf(A,node,modes[node],Q;orthonormal)
            cross[node] = Q'*b.leaves[node]
            overlap[node] = Q'*x.leaves[node]
        else
            C = cores[node]
            cores[node] = reshape(Matrix(transpose(Q)),rank,size(C,2),size(C,3))
            left,right = tree.children[node]
            up[node] = _env_up(A,node,_basis_matrix(cores[node]),cores[node],up[left],up[right];orthonormal,threaded)
            cross[node] = Q'*_basis_matrix(_project_children(b.cores[node],cross[left],cross[right]))
            overlap[node] = Q'*_basis_matrix(_project_children(x.cores[node],overlap[left],overlap[right]))
        end
        return nothing
    end
    function visit!(node,budget)
        # Full-rank leaves are skipped; the parent re-solve covers their space.
        _isleaf(tree,node) && _full_leaf_basis(leaves[node]) || solve!(node)
        if !_isleaf(tree,node)
            left,right = tree.children[node]
            entry = cores[node]
            active = (left!=time_leaf,right!=time_leaf)
            rhs = _parent_product(b.cores[node],rhsdown[node])
            frames = _etter_child_frames(entry,down[node],up[left],up[right],rhs;
                cross_left=cross[left],cross_right=cross[right],active,operator=A,node,threaded)
            # Publish inputs before spawning; descendants never write parent slots.
            for (child,frame) in zip((left,right),frames)
                if child == time_leaf
                    overlap[child] = leaves[child]'*x.leaves[child]
                    continue
                end
                down[child],rhsdown[child] = frame.operator,frame.rhs
                if _isleaf(tree,child)
                    leaves[child] = leaves[child]*transpose(frame.transfer)
                else
                    cores[child] = _parent_product(cores[child],frame.transfer)
                end
            end
            if all(active) && budget>1
                lb = clamp(round(Int,budget*sizes[left]/(sizes[left]+sizes[right])),1,budget-1)
                spawned[node] += 1
                task = Threads.@spawn try visit!(left,lb); nothing catch e; e end
                right_error = try visit!(right,budget-lb); nothing catch e; e end
                left_error = fetch(task)
                isnothing(left_error) || throw(left_error)
                isnothing(right_error) || throw(right_error)
            else
                active[1] && visit!(left,budget)
                active[2] && visit!(right,budget)
            end
            # Inverse-free connector reconstruction: project the pre-branch
            # tensor onto the new child bases, then solve the parent Galerkin
            # system. The solve's starting iterate differs from Algorithm 4.
            cores[node] = _project_children(entry,overlap[left],overlap[right])
            solve!(node)
        end
        node==tree.root || ascend!(node)
        return nothing
    end
    visit!(tree.root,threaded ? Threads.nthreads(:default) : 1)
    for node in tree.postorder
        if _isleaf(tree,node)
            x.leaves[node] = leaves[node]
        else
            x.cores[node] = cores[node]
        end
        isnothing(slots[node]) || (workspace.buffers[node] = slots[node])
    end
    workspace.buffer_allocations += sum(allocations)
    workspace.buffer_reuses += sum(reuses)
    return (local_solves=sum(solves),matvecs=sum(matvecs),iterations=sum(iterations),
            max_residual=maximum(residuals),environment_seconds,spawned_tasks=sum(spawned))
end
