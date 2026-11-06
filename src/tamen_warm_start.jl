"""
    TAMEnWarmStart(solution::SpaceTimeTTN, residual=nothing)

Owned initial-guess data for a subsequent interval. Copies all tensor arrays
and time nodes; topology is shared and read-only, as for TTNTensor. Treat fields
as read-only. Contains no operators, environments or convergence certificate.
Usually obtained with `tamen_step(...;store_warm_start=true)` as `info.warm_start`.
Different solves may read the same hint concurrently with separate workspaces.
"""
struct TAMEnWarmStart
    solution::SpaceTimeTTN
    residual::Union{Nothing,TTNTensor}
    function TAMEnWarmStart(solution::SpaceTimeTTN,residual::Union{Nothing,TTNTensor}=nothing)
        isnothing(residual) || _check_layout(solution.state,residual)
        snapshot = SpaceTimeTTN(copy(solution.state),copy(solution.initial),copy(solution.nodes),
                               solution.time_leaf,solution.time_scheme)
        new(snapshot,isnothing(residual) ? nothing : copy(residual))
    end
end

# Coefficients in the previous solution's spatial root basis. Its parent rank
# need not be one; do not construct a standalone TTNTensor with that root.
function _tamen_spatial_projection(solution,y)
    x,tree = solution.state,y.tree
    cross = Vector{Matrix{ComplexF64}}(undef,length(tree.children))
    for node in tree.postorder
        cross[node] = if _isleaf(tree,node)
            x.leaves[node]'*y.leaves[node]
        else
            left,right = tree.children[node]
            _basis_matrix(x.cores[node])'*
                _basis_matrix(_project_children(y.cores[node],cross[left],cross[right]))
        end
    end
    return vec(cross[tree.root])
end

_tamen_cold_guess(initial,nt,status) =
    (orthogonalize!(_time_lift(initial,ones(nt);full_time_basis=true)),nothing,status)

function _tamen_initial_guess(initial,grid,hint)
    nt = length(grid.d0)
    isnothing(hint) && return _tamen_cold_guess(initial,nt,:not_requested)
    hint isa TAMEnWarmStart || throw(ArgumentError("warm_start must be nothing or TAMEnWarmStart"))
    old = hint.solution
    a,b = initial.tree,old.initial.tree
    if a.root!=b.root || a.children!=b.children || a.leaf_order!=b.leaf_order ||
       leafdims(initial)!=leafdims(old.initial)
        return _tamen_cold_guess(initial,nt,:layout_mismatch)
    end
    endpoint = time_slice(old,length(old.nodes))
    scale = max(norm(endpoint),norm(initial))
    ttn_distance(endpoint,initial) <= 128eps(Float64)*scale ||
        return _tamen_cold_guess(initial,nt,:initial_mismatch)
    ratio = last(grid.nodes)/last(old.nodes)
    isfinite(ratio) || return _tamen_cold_guess(initial,nt,:duration_ratio)
    start = _tamen_spatial_projection(old,old.initial)
    anchor = _tamen_spatial_projection(old,initial)
    C = old.state.cores[old.state.tree.root]
    frame = reshape(C,size(C,2),size(C,3))*transpose(old.state.leaves[old.time_leaf])
    samples = hcat(start,old.time_scheme==:cheb ? frame : frame[:,2:end])
    oldnodes = old.nodes ./ last(old.nodes)
    newnodes = grid.scheme==:cheb ? grid.nodes[2:end] : grid.nodes
    guess = Matrix{ComplexF64}(undef,length(start),nt)
    for (j,t) in enumerate(newnodes)
        # Replay increments on a normalized interval, scaling their amplitude
        # with duration. The interpolation is always inside the old interval.
        theta = clamp(t/last(grid.nodes),0.,1.)
        weights = _time_interpolation(oldnodes,theta,old.time_scheme)
        guess[:,j] = anchor + ratio*(samples*weights-start)
    end
    all(isfinite,guess) || return _tamen_cold_guess(initial,nt,:nonfinite_guess)
    x = copy(old.state)
    x.leaves[old.time_leaf] = Matrix{ComplexF64}(I,nt,nt)
    x.cores[x.tree.root] = reshape(guess,1,size(guess,1),nt)
    orthogonalize!(x)
    # Residual values are merely initial guesses; all mixed environments and
    # local projections are rebuilt for the new A and b. A changed temporal
    # discretization uses a fresh auxiliary seed instead.
    compatible_residual = grid.scheme==old.time_scheme && length(old.nodes)==length(grid.nodes)
    residual = compatible_residual && !isnothing(hint.residual) ? copy(hint.residual) : nothing
    return x,residual,:used
end
