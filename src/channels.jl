# Exact sparse tree operator for tAMEn ALS environments.
#
# A SumOfProducts term t is split at every edge e (subtree S, complement S̄)
# into A_t (its factors in S) and B_t (its factors in S̄). The coefficient is
# attached to one support leaf κ(t): the support leaf touched by the fewest
# terms (a bath leaf rather than a shared system leaf). Each edge carries
#
#   :I      terms with no factor in S            up I        down Σ c B
#   :done   terms with every factor in S         up Σ c A    down I
#   (:A,A)  other terms with κ ∉ S, keyed by A   up A        down Σ c B
#   (:B,B)  other terms with κ ∈ S, keyed by B   up Σ c A    down B
#
# so every coefficient is applied exactly once, at κ. A node stores sparse
# transitions (α,β,γ,c): U_parent[α] = Σ c U_left[β] ⊗ U_right[γ]. Downward
# environments and local operators follow from the same transitions.
# Channels sharing a key are exact sums; no numerical compression is used.
# For HEOM, bath-subtree edges carry 2 + 2*(sites in the subtree) channels.
struct _ChannelOperator
    tree::BinaryTree
    dimensions::Vector{Int}
    channels::Vector{Int}
    leaf_ops::Vector{Vector{Union{Nothing,Matrix{ComplexF64}}}}
    transitions::Vector{Vector{Tuple{Int,Int,Int,ComplexF64}}}
end

const _ChannelKey = Tuple{Symbol,Vector{Pair{Int,Matrix{ComplexF64}}}}

function _channel_operator(op::SumOfProducts,tree::BinaryTree)
    length(tree.leaf_order) == length(op.dimensions) ||
        throw(DimensionMismatch("operator modes and tree leaves differ"))
    _isleaf(tree,tree.root) && throw(ArgumentError("channel operators need an internal root"))
    n = length(tree.children)
    modes = zeros(Int,n)
    for (mode,node) in enumerate(tree.leaf_order)
        modes[node] = mode
    end
    inside = Vector{BitSet}(undef,n)
    for node in tree.postorder
        inside[node] = _isleaf(tree,node) ? BitSet(modes[node]) :
            union(inside[tree.children[node][1]],inside[tree.children[node][2]])
    end
    terms = [t for t in op.terms if !iszero(t.coefficient)]
    # A zero operator keeps identity channels and one zero root transition.
    isempty(terms) && push!(terms,ProductTerm(0,Dict{Int,Matrix{ComplexF64}}()))
    supports = [sort!(collect(keys(t.factors))) for t in terms]
    degree = zeros(Int,length(op.dimensions))
    for support in supports, mode in support
        degree[mode] += 1
    end
    # 0 marks a scalar term; its coefficient is introduced at the root.
    anchors = [isempty(s) ? 0 : first(sort(s;by=m->(degree[m],m))) for s in supports]
    function key(k,node)
        S,support = inside[node],supports[k]
        hits = count(in(S),support)
        hits == 0 && return (:I,Pair{Int,Matrix{ComplexF64}}[])
        hits == length(support) && return (:done,Pair{Int,Matrix{ComplexF64}}[])
        factors = terms[k].factors
        anchors[k] in S && return (:B,[m=>factors[m] for m in support if !(m in S)])
        return (:A,[m=>factors[m] for m in support if m in S])
    end
    index = [Dict{_ChannelKey,Int}() for _ in 1:n]
    kinds = [Symbol[] for _ in 1:n]
    channel = ones(Int,length(terms),n)
    for node in 1:n
        node == tree.root && continue
        for k in eachindex(terms)
            entry = key(k,node)
            channel[k,node] = get!(index[node],entry) do
                push!(kinds[node],entry[1])
                length(kinds[node])
            end
        end
    end
    channels = [node == tree.root ? 1 : length(kinds[node]) for node in 1:n]
    leaf_ops = [Union{Nothing,Matrix{ComplexF64}}[] for _ in 1:n]
    for node in tree.leaf_order
        ops = Vector{Union{Nothing,Matrix{ComplexF64}}}(nothing,channels[node])
        for (k,t) in enumerate(terms)
            alpha = channel[k,node]
            kind = kinds[node][alpha]
            kind == :I && continue
            factor = t.factors[modes[node]]
            if kind == :A
                ops[alpha] = factor          # bare, identical across the channel
            else                             # :done/:B: this leaf is κ(t)
                scaled = t.coefficient .* factor
                ops[alpha] = isnothing(ops[alpha]) ? scaled : ops[alpha] .+ scaled
            end
        end
        leaf_ops[node] = ops
    end
    transitions = [Tuple{Int,Int,Int,ComplexF64}[] for _ in 1:n]
    for node in tree.postorder
        _isleaf(tree,node) && continue
        a,b = tree.children[node]
        carried = NTuple{3,Int}[]
        seen = Set{NTuple{3,Int}}()
        introduced = Dict{NTuple{3,Int},ComplexF64}()
        for k in eachindex(terms)
            triple = (channel[k,node],channel[k,a],channel[k,b])
            if anchors[k] == 0 && node == tree.root
                introduced[triple] = get(introduced,triple,zero(ComplexF64))+terms[k].coefficient
            elseif !(triple in seen)
                # Child channels already sum every term mapped to this triple.
                push!(seen,triple)
                push!(carried,triple)
            end
        end
        transitions[node] = [[(t...,one(ComplexF64)) for t in carried];
                             [(t...,c) for (t,c) in introduced]]
    end
    return _ChannelOperator(tree,copy(op.dimensions),channels,leaf_ops,transitions)
end

function _check_operator(operator::_ChannelOperator,state::TTNTensor)
    operator.dimensions == leafdims(state) && operator.tree.children == state.tree.children &&
        operator.tree.leaf_order == state.tree.leaf_order && operator.tree.root == state.tree.root ||
        throw(DimensionMismatch("channel operator was built for a different tree or layout"))
end

function _channel_leaf_envs(P,node,bra,ket=bra;orthonormal=false)
    return map(P.leaf_ops[node]) do O
        if isnothing(O)
            orthonormal && bra === ket ? Matrix{ComplexF64}(I,size(bra,2),size(bra,2)) : bra'*ket
        else
            bra'*(O*ket)
        end
    end
end

# out[t] = Σ_{entries (t,i,j,c)} c*f(X[i],Y[j]) for bilinear f. Entries that
# share one factor are summed in the other first, so f runs once per group.
# f maps factor lists to results that may alias each other (read-only).
function _grouped_bilinear(f,ntargets,entries,X,Y)
    bytarget = [Int[] for _ in 1:ntargets]
    for (e,entry) in enumerate(entries)
        push!(bytarget[entry[1]],e)
    end
    Ls,Rs = Matrix{ComplexF64}[],Matrix{ComplexF64}[]
    owners,scales = Int[],ComplexF64[]
    for target in 1:ntargets
        members = bytarget[target]
        isempty(members) && continue
        byx = length(unique(entries[e][2] for e in members)) <= length(unique(entries[e][3] for e in members))
        shared,summed = byx ? (2,3) : (3,2)
        groups = Dict{Int,Vector{Int}}()
        order = Int[]
        for e in members
            g = entries[e][shared]
            haskey(groups,g) || push!(order,g)
            push!(get!(groups,g,Int[]),e)
        end
        for g in order
            group = groups[g]
            source = summed == 3 ? Y : X
            if length(group) == 1
                e = only(group)
                other = source[entries[e][summed]]
                scale = entries[e][4]
            else
                other = zeros(ComplexF64,size(source[entries[first(group)][summed]]))
                for e in group
                    other .+= entries[e][4] .* source[entries[e][summed]]
                end
                scale = one(ComplexF64)
            end
            fixed = (shared == 2 ? X : Y)[g]
            push!(Ls,shared == 2 ? fixed : other)
            push!(Rs,shared == 2 ? other : fixed)
            push!(owners,target)
            push!(scales,scale)
        end
    end
    results = isempty(Ls) ? Matrix{ComplexF64}[] : f(Ls,Rs)
    out = Vector{Matrix{ComplexF64}}(undef,ntargets)
    count = zeros(Int,ntargets)
    foreach(t->count[t]+=1,owners)
    started = falses(ntargets)
    for (g,target) in enumerate(owners)
        value = results[g]
        if count[target] == 1
            out[target] = isone(scales[g]) ? value : scales[g] .* value
        elseif !started[target]
            out[target] = scales[g] .* value
            started[target] = true
        else
            out[target] .+= scales[g] .* value
        end
    end
    for target in 1:ntargets
        isassigned(out,target) || throw(ArgumentError("channel $target has no transition"))
    end
    return out
end

function _channel_up(P,node,bra,core,left,right;orthonormal=false,threaded=false)
    return _grouped_bilinear(P.channels[node],P.transitions[node],left,right) do Ls,Rs
        _mixed_subtree_operators(bra,core,Ls,Rs;orthonormal,threaded)
    end
end

function _complement_source(Q,boundary,sibling)
    b,s = size(first(boundary),1),size(first(sibling),1)
    return permutedims(reshape(Q,b,s,:),(3,1,2))
end

function _channel_down(P,parent,mode,Q,source,boundary,sibling;orthonormal=false,threaded=false)
    child = P.tree.children[parent][mode-1]
    entries = mode == 2 ? [(b,a,c,x) for (a,b,c,x) in P.transitions[parent]] :
                          [(c,a,b,x) for (a,b,c,x) in P.transitions[parent]]
    return _grouped_bilinear(P.channels[child],entries,boundary,sibling) do Bs,Ss
        _mixed_subtree_operators(Q,source,Bs,Ss;orthonormal,threaded)
    end
end

function _channel_core_terms(P,node,boundary,left,right)
    terms = [_CoreOperatorTerm(c,(boundary[a],left[b],right[g])) for (a,b,g,c) in P.transitions[node]]
    return _combine_all_core_terms(terms)
end

function _channel_leaf_workspace(P,node,boundary,initial;buffers=nothing)
    local_product,acted = isnothing(buffers) ? (similar(initial),similar(initial)) : buffers
    ops = P.leaf_ops[node]
    transposed = transpose.(boundary)
    identity_boundary = isone.(boundary)
    function action(x)
        result = zeros(ComplexF64,size(x))
        for (alpha,O) in enumerate(ops)
            source = x
            if !isnothing(O)
                mul!(local_product,O,x)
                source = local_product
            end
            if !identity_boundary[alpha]
                mul!(acted,source,transposed[alpha])
                source = acted
            end
            result .+= source
        end
        return result
    end
    return action
end

# Operator-generic environment layer used by tAMEn sweeps and local residuals.
# The SoP path keeps its historical Gram matrices in every frame. The channel
# path uses exact identities only where the caller asserts orthonormal frames.
_assume_orthonormal(::SumOfProducts) = false
_assume_orthonormal(::_ChannelOperator) = true
_root_boundary(A::SumOfProducts) = [ones(ComplexF64,1,1) for _ in A.terms]
_root_boundary(::_ChannelOperator) = [ones(ComplexF64,1,1)]
_env_leaf(A::SumOfProducts,node,mode,bra,ket=bra;orthonormal=false) =
    _leaf_operators(A,mode,bra,ket;orthonormal)
_env_leaf(A::_ChannelOperator,node,mode,bra,ket=bra;orthonormal=false) =
    _channel_leaf_envs(A,node,bra,ket;orthonormal)
_env_up(::SumOfProducts,node,bra,core,left,right;orthonormal=false,threaded=false) =
    _mixed_subtree_operators(bra,core,left,right;orthonormal,threaded)
_env_up(A::_ChannelOperator,node,bra,core,left,right;orthonormal=false,threaded=false) =
    _channel_up(A,node,bra,core,left,right;orthonormal,threaded)
function _env_down(::SumOfProducts,parent,mode,Q,boundary,sibling;source=nothing,orthonormal=false,threaded=false)
    isnothing(source) && return _complement_operators(Q,boundary,sibling;orthonormal,threaded)
    return _mixed_subtree_operators(Q,source,boundary,sibling;orthonormal,threaded)
end
function _env_down(A::_ChannelOperator,parent,mode,Q,boundary,sibling;source=nothing,orthonormal=false,threaded=false)
    isempty(boundary) && return Matrix{ComplexF64}[]
    frame = isnothing(source) ? _complement_source(Q,boundary,sibling) : source
    return _channel_down(A,parent,mode,Q,frame,boundary,sibling;orthonormal,threaded)
end
_leaf_action(A::SumOfProducts,node,mode,boundary,initial;buffers=nothing) =
    _leaf_rhs_workspace(A,mode,boundary,initial;buffers)
_leaf_action(A::_ChannelOperator,node,mode,boundary,initial;buffers=nothing) =
    _channel_leaf_workspace(A,node,boundary,initial;buffers)
_core_terms(A::SumOfProducts,node,boundary,left,right) = _group_core_terms(A,boundary,left,right)
_core_terms(A::_ChannelOperator,node,boundary,left,right) = _channel_core_terms(A,node,boundary,left,right)
_core_action(A::SumOfProducts,node,boundary,left,right,initial;buffers=nothing,threaded=false) =
    _core_terms_workspace(_core_terms(A,node,boundary,left,right),initial;buffers)

# Large channel cores split their terms into a fixed number of groups (decided
# by shape and term count only), solved concurrently when `threaded`. Groups
# add cost for small cores, so they stay single.
# Internal tuning knobs (Refs so tests can force grouping on small problems).
const _CORE_CHUNKS = Ref(4)
const _CORE_CHUNK_MIN_SIZE = Ref(4096)
const _CORE_CHUNK_MIN_TERMS = Ref(4)
const _CORE_LOWRANK = Ref(true)
# Factor SVDs only pay off for large cores (measured on Holstein HEOM).
const _CORE_LOWRANK_MIN_SIZE = Ref(131072)
function _core_action(A::_ChannelOperator,node,boundary,left,right,initial;buffers=nothing,threaded=false)
    terms = _core_terms(A,node,boundary,left,right)
    chunks = length(initial) >= _CORE_CHUNK_MIN_SIZE[] && length(terms) >= _CORE_CHUNK_MIN_TERMS[] ? _CORE_CHUNKS[] : 1
    lowrank = _CORE_LOWRANK[] && length(initial) >= _CORE_LOWRANK_MIN_SIZE[]
    return _core_terms_workspace(terms,initial;buffers,chunks,threaded,lowrank)
end

function _leaf_subtract!(rhs,A::SumOfProducts,node,mode,boundary,x)
    for (k,term) in enumerate(A.terms)
        iszero(term.coefficient) && continue
        rhs .-= term.coefficient .* (_local_action(term,mode,x)*transpose(boundary[k]))
    end
    return rhs
end
function _leaf_subtract!(rhs,A::_ChannelOperator,node,mode,boundary,x)
    for (alpha,O) in enumerate(A.leaf_ops[node])
        rhs .-= (isnothing(O) ? x : O*x)*transpose(boundary[alpha])
    end
    return rhs
end

# Generator projected onto the spatial root basis for the time-error estimate.
# `spatial` is the root child holding space; the other child is the time leaf,
# whose environments are identities for every spatial term.
function _projected_spatial_generator(A::SumOfProducts,up,spatial,root,rank)
    L = zeros(ComplexF64,rank,rank)
    for (k,term) in enumerate(A.terms)
        L .+= term.coefficient .* up[spatial][k]
    end
    return L
end
function _projected_spatial_generator(A::_ChannelOperator,up,spatial,root,rank)
    first_child = A.tree.children[root][1] == spatial
    L = zeros(ComplexF64,rank,rank)
    for (_,b,g,c) in A.transitions[root]
        L .+= c .* up[spatial][first_child ? b : g]
    end
    return L
end

_local_operator(A::SumOfProducts,tree,choice) =
    choice == :channels ? _channel_operator(A,tree) : A

function _check_local_operator(choice)
    choice in (:sop,:channels) || throw(ArgumentError("local_operator must be :sop or :channels"))
end
