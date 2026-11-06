# Integrated defect on a finer temporal grid in the final spatial space.
# No second spatial ALS solve, full tensor, or global generator is needed.
function _tamen_time_error(solution,generator;threaded=false)
    x = solution.state
    spatial_root = solution.initial.tree.root
    nt = size(x.leaves[solution.time_leaf],1)
    op = SumOfProducts([generator.dimensions;nt],generator.terms)
    initial_lift = _time_lift(solution.initial,ones(nt))
    up,cross = _tamen_upward(op,x,initial_lift;threaded)
    C = x.cores[x.tree.root]
    rank = size(C,2)
    L = zeros(ComplexF64,rank,rank)
    for (k,term) in enumerate(generator.terms)
        L .+= term.coefficient .* up[spatial_root][k]
    end
    u0 = vec(cross[spatial_root])
    projection_error = ttn_distance(solution.initial,_spatial_coefficients(solution,u0))
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
