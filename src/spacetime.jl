"""
    chebyshev_collocation(time_points, duration)

Increasing Chebyshev–Lobatto nodes on `[0,duration]`. Eliminate the initial
value from the differentiation matrix: `(D ⊗ I - I ⊗ L)*vec(X) = d0 ⊗ x0`,
with the spatial axes fastest. `time_points` counts unknown samples, including the endpoint but
excluding the initial point. Returns `nodes`, `D`, and `d0`.
"""
function chebyshev_collocation(time_points, duration)
    time_points isa Integer && time_points >= 1 ||
        throw(ArgumentError("time_points must be a positive integer"))
    duration isa Real && isfinite(duration) && duration > 0 &&
        isfinite(Float64(duration)) && Float64(duration) > 0 ||
        throw(ArgumentError("duration must be finite and positive in Float64"))
    n, h = Int(time_points), Float64(duration)
    nodes = [h*sinpi(j/(2n))^2 for j in 0:n]
    nodes[1], nodes[end] = 0.0, h
    all(diff(nodes) .> 0) || throw(ArgumentError("collocation nodes are not distinct"))
    weights = [(-1.0)^j * (j in (0,n) ? 0.5 : 1.0) for j in 0:n]
    full = zeros(Float64,n+1,n+1)
    for i in 1:n+1, j in 1:n+1
        i == j && continue
        full[i,j] = weights[j] / (weights[i]*(nodes[i]-nodes[j]))
    end
    for i in 1:n+1
        full[i,i] = -sum(view(full,i,:))
    end
    all(isfinite,full) || throw(ArgumentError("duration produces nonfinite differentiation matrix"))
    return (; nodes, D=full[2:end,2:end], d0=-full[2:end,1])
end

"""A fixed-interval TTN collocation solution. Use `time_slice` to extract a sample.
`nodes` includes the initial time zero. The final physical axis stores the
unknown samples (Chebyshev), or all samples including the initial equation
(CN). Treat stored tensors and arrays as read-only.
"""
struct SpaceTimeTTN
    state::TTNTensor
    initial::TTNTensor
    nodes::Vector{Float64}
    time_leaf::Int
    time_scheme::Symbol
end
SpaceTimeTTN(state,initial,nodes,time_leaf) = SpaceTimeTTN(state,initial,nodes,time_leaf,:cheb)

function _time_scheme(value)
    value isa Union{Symbol,AbstractString} || throw(ArgumentError("time_scheme must be :cheb or :cn"))
    scheme = Symbol(lowercase(String(value)))
    scheme in (:cheb,:cn) || throw(ArgumentError("time_scheme must be :cheb or :cn"))
    return scheme
end

function _tamen_grid(nt,h,scheme)
    scheme = _time_scheme(scheme)
    if scheme == :cheb
        grid = chebyshev_collocation(nt,h)
        return merge(grid,(;M=Matrix{Float64}(I,nt,nt),scheme))
    end
    nt isa Integer && nt >= 3 && isodd(nt) ||
        throw(ArgumentError("CN requires an odd time_points >= 3, including the initial point"))
    # Reuse duration validation and check the representability of every node.
    chebyshev_collocation(1,h)
    nodes = collect(range(0.0,Float64(h);length=nt))
    all(diff(nodes).>0) || throw(ArgumentError("CN nodes are not distinct"))
    D = Matrix{Float64}(I,nt,nt)
    M = zeros(Float64,nt,nt)
    for j in 2:nt
        D[j,j-1] = -1
        M[j,j-1] = M[j,j] = (nodes[j]-nodes[j-1])/2
    end
    d0 = zeros(Float64,nt); d0[1] = 1
    return (;nodes,D,M,d0,scheme)
end

function _time_lift(x, time_vector; full_time_basis=false)
    n = length(x.tree.children)
    tree = BinaryTree([x.tree.children; (0,0); (x.tree.root,n+1)];
                      root=n+2,leaf_order=[x.tree.leaf_order;n+1])
    nt = length(time_vector)
    leaves = Dict(k=>copy(v) for (k,v) in x.leaves)
    cores = Dict(k=>copy(v) for (k,v) in x.cores)
    leaves[n+1] = full_time_basis ? Matrix{ComplexF64}(I,nt,nt) :
                                  reshape(ComplexF64.(time_vector),nt,1)
    cores[n+2] = full_time_basis ? reshape(ComplexF64.(time_vector),1,1,nt) :
                                  ones(ComplexF64,1,1,1)
    return TTNTensor(tree,leaves,cores)
end

function _spacetime_operator(op, grid)
    time_mode = length(op.dimensions)+1
    terms = [ProductTerm(1,Dict(time_mode=>grid.D))]
    for t in op.terms
        factors = copy(t.factors)
        # Chebyshev's M is the identity: leave the time factor implicit so
        # identity environments and channels stay exact (same operator).
        hasproperty(grid,:M) && !isone(grid.M) && (factors[time_mode] = grid.M)
        push!(terms,ProductTerm(-t.coefficient,factors))
    end
    return SumOfProducts([op.dimensions;length(grid.d0)],terms)
end

# Reset a complete orthonormal time basis to the identity without changing X.
function _identity_time!(x, time_leaf)
    U = x.leaves[time_leaf]
    size(U,1) == size(U,2) || error("time basis lost columns")
    parent = x.tree.parent[time_leaf]
    mode = x.tree.children[parent][1] == time_leaf ? 2 : 3
    x.cores[parent] = _child_product(x.cores[parent],U,mode)
    x.leaves[time_leaf] = Matrix{ComplexF64}(I,size(U,1),size(U,1))
    return x
end

"""
    time_slice(solution::SpaceTimeTTN, index)

Extract an independent spatial TTN at `solution.nodes[index]`, including the
initial point at index 1. Only the time leaf and root are contracted; the
full spatial tensor is never expanded. No interpolation is performed.
"""
function time_slice(solution::SpaceTimeTTN, index::Integer)
    checkbounds(solution.nodes,index)
    index == 1 && return copy(solution.initial)
    st = solution.state
    C = st.cores[st.tree.root]
    sample = solution.time_scheme == :cheb ? index-1 : index
    time_row = view(st.leaves[solution.time_leaf],sample,:)
    weights = reshape(C,size(C,2),size(C,3)) * time_row
    return _spatial_coefficients(solution,weights)
end

# dot(c, time_slice(solution, j)) for every observable c and sample j from one
# contraction per c: the spatial basis keeps its root bond open, and each
# sample weights that bond with its root/time row (as in time_slice).
function _invariant_history(solution::SpaceTimeTTN,cs)
    nodes = length(solution.nodes)
    history = zeros(ComplexF64,length(cs),nodes)
    isempty(cs) && return history
    st,original = solution.state,solution.initial.tree
    C = st.cores[st.tree.root]
    offset = solution.time_scheme == :cheb ? 1 : 0
    samples = [index-offset for index in 2:nodes]
    weights = reshape(C,size(C,2),size(C,3)) * transpose(st.leaves[solution.time_leaf][samples,:])
    for (i,c) in enumerate(cs)
        envs = Dict{Int,Matrix{ComplexF64}}()
        for node in original.postorder
            if _isleaf(original,node)
                envs[node] = c.leaves[node]' * st.leaves[node]
            else
                left,right = original.children[node]
                projected = _child_product(_child_product(st.cores[node],envs[left],2),envs[right],3)
                ccore = c.cores[node]
                envs[node] = conj.(reshape(ccore,size(ccore,1),:)) * transpose(reshape(projected,size(projected,1),:))
            end
        end
        history[i,1] = dot(c,solution.initial)
        nodes > 1 && (history[i,2:end] = vec(envs[original.root]*weights))
    end
    return history
end

function _spatial_coefficients(solution,weights)
    st,original = solution.state,solution.initial.tree
    leaves = Dict(node=>copy(st.leaves[node]) for node in original.leaf_order)
    cores = Dict(node=>copy(st.cores[node]) for node in original.postorder
                 if !_isleaf(original,node))
    if _isleaf(original,original.root)
        leaves[original.root] = leaves[original.root] * reshape(weights,:,1)
    else
        cores[original.root] = _parent_product(cores[original.root],transpose(weights))
    end
    return TTNTensor(original,leaves,cores)
end

function _time_interpolation(nodes,t,scheme)
    weights = zeros(Float64,length(nodes))
    exact = findfirst(==(t),nodes)
    if !isnothing(exact)
        weights[exact] = 1
    elseif scheme == :cn
        j = searchsortedlast(nodes,t)
        theta = (t-nodes[j])/(nodes[j+1]-nodes[j])
        weights[j],weights[j+1] = 1-theta,theta
    else
        # Common scaling cancels in the barycentric ratio and prevents 1/d
        # from overflowing at a valid subnormal distance from a time node.
        nearest = minimum(abs(t-node) for node in nodes)
        for j in eachindex(nodes)
            weights[j] = (-1.0)^(j-1)*(j in (1,length(nodes)) ? 0.5 : 1)*(nearest/(t-nodes[j]))
        end
        weights ./= sum(weights)
    end
    return weights
end

"""
    time_snapshot(solution, time)

Independent spatial TTN at a relative time in the solution interval. Uses
barycentric polynomial interpolation for Chebyshev or piecewise linear
interpolation for CN, with the original initial value. No extrapolation,
normalization or spatial truncation is performed.
"""
function time_snapshot(solution::SpaceTimeTTN,t::Real)
    isfinite(t) && first(solution.nodes)<=t<=last(solution.nodes) ||
        throw(ArgumentError("time must be finite and inside the solution interval"))
    time = Float64(t)
    exact = findfirst(==(time),solution.nodes)
    !isnothing(exact) && return time_slice(solution,exact)
    w = _time_interpolation(solution.nodes,time,solution.time_scheme)
    C = solution.state.cores[solution.state.tree.root]
    U = solution.state.leaves[solution.time_leaf]
    frame = reshape(C,size(C,2),size(C,3))*transpose(U)
    active = solution.time_scheme == :cheb ? frame : view(frame,:,2:size(frame,2))
    x = _spatial_coefficients(solution,active*w[2:end])
    return iszero(w[1]) ? x : _tamen_add(x,solution.initial,w[1])
end

# Exact binary block sum. Compress after each addition, never form a block
# sum of every SoP term at once. Coefficients live only at the root.
function _tamen_add(x,y,alpha=1)
    _check_layout(x,y)
    tree = x.tree
    if _isleaf(tree,tree.root)
        return TTNTensor(tree,Dict(tree.root=>x.leaves[tree.root]+alpha*y.leaves[tree.root]),
                         Dict{Int,Array{ComplexF64,3}}())
    end
    leaves = Dict(n=>hcat(x.leaves[n],y.leaves[n]) for n in tree.leaf_order)
    cores = Dict{Int,Array{ComplexF64,3}}()
    for n in tree.postorder
        _isleaf(tree,n) && continue
        a,b = x.cores[n],y.cores[n]
        pa,la,ra = size(a)
        pb,lb,rb = size(b)
        root = n == tree.root
        C = zeros(ComplexF64,root ? 1 : pa+pb,la+lb,ra+rb)
        C[1:pa,1:la,1:ra] .= a
        C[root ? (1:1) : (pa+1:pa+pb),la+1:la+lb,ra+1:ra+rb] .= root ? alpha*b : b
        cores[n] = C
    end
    return TTNTensor(tree,leaves,cores)
end

