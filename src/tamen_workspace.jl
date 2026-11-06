"""
    TAMEnWorkspace()

Scratch storage reusable across sequential `tamen_step`/`tamen_evolve` calls.
Owns one current ALS buffer set per node, growing per-node residual projection/QR
buffers, reusable serial weighted-cut scratch, and unit-interval time grids.
Residual buffers retain their largest used
capacity until the workspace is discarded. Environment
values are rebuilt after every basis change and are never cached across solves.
Concurrent calls need distinct workspaces; simultaneous/reentrant use of one
workspace throws. A lease is released even if the solve or observer fails.
"""
mutable struct TAMEnWorkspace
    busy::Threads.Atomic{Bool}
    buffers::Dict{Int,Any}
    grids::Dict{Tuple{Int,Symbol},NamedTuple}
    buffer_allocations::Int
    buffer_reuses::Int
    grid_hits::Int
    certification::_ResidualWorkspace
end
TAMEnWorkspace() = TAMEnWorkspace(Threads.Atomic{Bool}(false),Dict{Int,Any}(),
    Dict{Tuple{Int,Symbol},NamedTuple}(),0,0,0,_ResidualWorkspace())

function _with_tamen_workspace(f,workspace)
    work = isnothing(workspace) ? TAMEnWorkspace() : workspace
    work isa TAMEnWorkspace || throw(ArgumentError("workspace must be a TAMEnWorkspace"))
    Threads.atomic_cas!(work.busy,false,true) &&
        throw(ArgumentError("TAMEnWorkspace is already in use; concurrent solves need separate workspaces"))
    try
        return f(work)
    finally
        Threads.atomic_xchg!(work.busy,false)
    end
end

_tamen_work_counts(w) = merge((;buffer_allocations=w.buffer_allocations,
    buffer_reuses=w.buffer_reuses,grid_hits=w.grid_hits),_residual_work_counts(w.certification))
function _tamen_work_delta(w,before)
    after = _tamen_work_counts(w)
    return NamedTuple{keys(after)}(values(after).-values(before))
end

function _tamen_grid!(w,nt,h,scheme)
    scheme = _time_scheme(scheme)
    # Validate nt and h even when reusing the unit-interval grid.
    nt isa Integer && nt>=1 && nt<=typemax(Int) || throw(ArgumentError("time_points must be a positive Int"))
    chebyshev_collocation(1,h)
    key = (Int(nt),scheme)
    if haskey(w.grids,key)
        w.grid_hits += 1
    else
        w.grids[key] = _tamen_grid(nt,1.0,scheme)
    end
    base = w.grids[key]
    scale = Float64(h)
    nodes = base.nodes*scale
    all(diff(nodes).>0) || throw(ArgumentError("time nodes are not distinct"))
    D = scheme == :cheb ? base.D/scale : copy(base.D)
    M = scheme == :cheb ? copy(base.M) : base.M*scale
    d0 = scheme == :cheb ? base.d0/scale : copy(base.d0)
    all(isfinite,D) && all(isfinite,d0) || throw(ArgumentError("time matrix overflow"))
    return (;nodes,D,M,d0,scheme)
end

function _tamen_buffers!(work,node,initial)
    signature = size(initial)
    cached = get(work.buffers,node,nothing)
    if !isnothing(cached) && cached.signature == signature
        work.buffer_reuses += 1
        return cached.buffers
    end
    buffers = _new_tamen_buffers(initial)
    work.buffers[node] = (;signature,buffers)
    work.buffer_allocations += 1
    return buffers
end

function _new_tamen_buffers(initial)
    if ndims(initial) == 2
        buffers = (similar(initial),similar(initial))
    else
        p,l,r = size(initial)
        buffers = (Array{ComplexF64}(undef,l,p,r),Array{ComplexF64}(undef,l,p,r),
                   Array{ComplexF64}(undef,r,p,l),Array{ComplexF64}(undef,r,p,l),
                   similar(initial),similar(initial))
    end
    return buffers
end

# Disjoint output slots, immutable inputs, per-call contraction buffers. Each
# parent waits for both children before reading their messages. Never mutate
# TTN Dicts or gauge centers from these tasks. Both branches join on failure.
function _tamen_upward(A,x,b;threaded=false)
    tree = x.tree
    # Non-root bases are orthonormal whenever sweeps call this (after prepare).
    orthonormal = _assume_orthonormal(A)
    up = Vector{Vector{Matrix{ComplexF64}}}(undef,length(tree.children))
    cross = Vector{Matrix{ComplexF64}}(undef,length(tree.children))
    modes = Dict(node=>mode for (mode,node) in enumerate(tree.leaf_order))
    sizes = ones(Int,length(tree.children))
    for node in tree.postorder
        _isleaf(tree,node) || (sizes[node] += sum(sizes[c] for c in tree.children[node]))
    end
    function visit(node,budget)
        if _isleaf(tree,node)
            up[node] = _env_leaf(A,node,modes[node],x.leaves[node];orthonormal)
            cross[node] = x.leaves[node]'*b.leaves[node]
            return
        end
        left,right = tree.children[node]
        if budget > 1
            lb = clamp(round(Int,budget*sizes[left]/(sizes[left]+sizes[right])),1,budget-1)
            task = Threads.@spawn try visit(left,lb); nothing catch e; e end
            right_error = try visit(right,budget-lb); nothing catch e; e end
            left_error = fetch(task)
            isnothing(left_error) || throw(left_error)
            isnothing(right_error) || throw(right_error)
        else
            visit(left,1)
            visit(right,1)
        end
        up[node] = _env_up(A,node,_basis_matrix(x.cores[node]),x.cores[node],up[left],up[right];
            orthonormal=orthonormal && node != tree.root)
        cross[node] = _basis_matrix(x.cores[node])'*
            _basis_matrix(_project_children(b.cores[node],cross[left],cross[right]))
        return
    end
    visit(tree.root,threaded ? Threads.nthreads(:default) : 1)
    return up,cross
end
