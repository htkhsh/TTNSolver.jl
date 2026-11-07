# Local generator terms can share two of their three factors. Sum the third
# factor exactly (no comparison tolerance or rank approximation), once before
# Arnoldi. Environment matrices are read-only for the lifetime of this solve.
struct _CoreOperatorTerm
    coefficient::ComplexF64
    factors::NTuple{3,Matrix{ComplexF64}}
end

function _combine_core_terms(terms, axis)
    others = filter(!=(axis), 1:3)
    lookup = Dict{Tuple{Matrix{ComplexF64},Matrix{ComplexF64}},Int}()
    groups = Vector{Int}[]
    for (i, term) in enumerate(terms)
        key = (term.factors[others[1]], term.factors[others[2]])
        index = get!(lookup, key) do
            push!(groups, Int[])
            length(groups)
        end
        push!(groups[index], i)
    end
    result = _CoreOperatorTerm[]
    for indices in groups
        first_term = terms[first(indices)]
        if length(indices) == 1
            push!(result, first_term)
            continue
        end
        summed = zeros(ComplexF64, size(first_term.factors[axis]))
        for i in indices
            summed .+= terms[i].coefficient .* terms[i].factors[axis]
        end
        factors = ntuple(i -> i == axis ? summed : first_term.factors[i], 3)
        push!(result, _CoreOperatorTerm(1, factors))
    end
    return result
end

function _group_core_terms(operator, boundary, left, right)
    terms = [_CoreOperatorTerm(t.coefficient, (boundary[k], left[k], right[k]))
             for (k,t) in enumerate(operator.terms) if !iszero(t.coefficient)]
    return _combine_all_core_terms(terms)
end

function _combine_all_core_terms(terms)
    while true
        count = length(terms)
        for axis in (2,3,1)
            terms = _combine_core_terms(terms, axis)
        end
        length(terms) == count && return terms
    end
end

# Scratch storage belongs to one sequential local Krylov solve. Never return
# a scratch array: Arnoldi retains action outputs while applying the operator
# again. Grouping may change summation roundoff, not the represented operator.
function _leaf_rhs_workspace(operator, mode, boundary, initial; buffers=nothing)
    local_product,acted = isnothing(buffers) ? (similar(initial),similar(initial)) : buffers
    transposed = transpose.(boundary)
    identity_boundary = isone.(boundary)
    function action(x)
        result = zeros(ComplexF64,size(x))
        for (k,term) in enumerate(operator.terms)
            iszero(term.coefficient) && continue
            if haskey(term.factors,mode)
                mul!(local_product,term.factors[mode],x)
                source = local_product
            else
                source = x
            end
            if !identity_boundary[k]
                mul!(acted,source,transposed[k])
                source = acted
            end
            result .+= term.coefficient .* source
        end
        return result
    end
    return action
end

_core_rhs_workspace(operator, boundary, left, right, initial; buffers=nothing) =
    _core_terms_workspace(_group_core_terms(operator, boundary, left, right), initial; buffers)

# Scratch for one group of local terms. Shared: x permuted for small parents.
_core_group_scratch(p,l,r) = (Array{ComplexF64}(undef,p,l,r),Array{ComplexF64}(undef,l,p,r),
    Array{ComplexF64}(undef,p,l,r),Array{ComplexF64}(undef,p,l,r),Array{ComplexF64}(undef,p,l,r))

# Mode products of a 3-index tensor into preallocated outputs.
function _mode1!(out,M,T)
    _,a,b = size(T)
    mul!(reshape(out,size(M,1),a*b),M,reshape(T,size(T,1),a*b))
    return out
end
function _mode2!(out,M,T)
    Mt = transpose(M)
    for k in axes(T,3)
        mul!(view(out,:,:,k),view(T,:,:,k),Mt)
    end
    return out
end
function _mode3!(out,M,T)
    p,a,b = size(T)
    mul!(reshape(out,p*a,size(M,1)),reshape(T,p*a,b),transpose(M))
    return out
end

# Child factors with low numerical rank k (singular values above
# _LOWRANK_RTOL*σ₁, k <= n/3) are applied as P*(Qᴴ*x): the mode is compressed
# first, every other factor acts on the smaller tensor, and P restores it.
# For HEOM the system-side channels V_j have rank (sites) in a space of
# dimension up to sites², which makes the system junction node much cheaper.
# Discarded singular values are below 1e-13σ₁, i.e. at roundoff level.
const _LOWRANK_RTOL = 1e-13
function _lowrank_factor(M)
    n = size(M,1)
    n >= 32 || return nothing
    F = svd(M)
    iszero(first(F.S)) && return nothing
    k = count(>(_LOWRANK_RTOL*F.S[1]),F.S)
    3k <= n || return nothing
    return (P=F.U[:,1:k] .* transpose(F.S[1:k]),Qh=F.Vt[1:k,:])
end

# Steps (mode, matrix, output dims) for one term, or nothing when the plain
# path is not at least 30% more expensive (estimated multiply-adds).
function _lowrank_plan(term,(bi,li,ri),p,l,r)
    B,L,R = term.factors
    lowL = li ? nothing : _lowrank_factor(L)
    lowR = ri ? nothing : _lowrank_factor(R)
    isnothing(lowL) && isnothing(lowR) && return nothing
    plain = (li ? 0 : l*l*p*r) + (ri ? 0 : r*r*p*l) + (bi ? 0 : p*p*l*r)
    steps = Tuple{Int,Matrix{ComplexF64},NTuple{3,Int}}[]
    cost = 0; a,b = l,r
    add!(mode,M) = begin
        dims = mode == 1 ? (size(M,1),a,b) : mode == 2 ? (p,size(M,1),b) : (p,a,size(M,1))
        cost += mode == 1 ? size(M,1)*p*a*b : mode == 2 ? size(M,1)*a*p*b : size(M,1)*b*p*a
        push!(steps,(mode,M,dims))
        mode == 2 && (a = size(M,1)); mode == 3 && (b = size(M,1))
    end
    isnothing(lowL) || add!(2,lowL.Qh)
    isnothing(lowR) || add!(3,lowR.Qh)
    isnothing(lowL) && !li && add!(2,L)
    isnothing(lowR) && !ri && add!(3,R)
    bi || add!(1,B)
    isnothing(lowL) || add!(2,lowL.P)
    isnothing(lowR) || add!(3,lowR.P)
    10cost <= 7plain || return nothing
    return steps
end

# dest .+= Σ coefficient * B ⊗ L ⊗ R x over `terms`, without permuting x: the
# right factor acts on a (p*l,r) view, the boundary on (p,l*r), and the left
# factor per right-index slice (or, for small p, on x_perm = permuted x).
# `lowrank=true` uses _lowrank_plan where it pays off.
function _core_terms_kernel(terms,p,l,r,scratch;lowrank=false)
    identities = [isone.(t.factors) for t in terms]
    transposed = [(transpose(t.factors[2]),transpose(t.factors[3])) for t in terms]
    left_out,left_tmp,right_out,acted = scratch
    right_mat,acted_mat = reshape(right_out,p*l,r),reshape(acted,p,l*r)
    slices = _slice_mode2(p,l)
    plans = [lowrank ? _lowrank_plan(t,ids,p,l,r) : nothing for (t,ids) in zip(terms,identities)]
    # Intermediate outputs are pooled by (step, dims): terms run in sequence,
    # so equal shapes can share storage. The last step writes into `acted`.
    pool = Dict{Tuple{Int,NTuple{3,Int}},Array{ComplexF64,3}}()
    outputs = map(plans) do plan
        isnothing(plan) && return nothing
        [k == length(plan) ? acted :
         get!(()->Array{ComplexF64}(undef,dims...),pool,(k,dims)) for (k,(_,_,dims)) in enumerate(plan)]
    end
    return function (dest,x,x_perm)
        for (i,(term,(bi,li,ri),(Lt,Rt))) in enumerate(zip(terms,identities,transposed))
            plan = plans[i]
            if !isnothing(plan)
                source = x
                for ((mode,M,_),out) in zip(plan,outputs[i])
                    source = mode == 1 ? _mode1!(out,M,source) : mode == 2 ? _mode2!(out,M,source) : _mode3!(out,M,source)
                end
                dest .+= term.coefficient .* source
                continue
            end
            source = x
            if !li
                if slices
                    for k in 1:r
                        mul!(view(left_out,:,:,k),view(source,:,:,k),Lt)
                    end
                else
                    mul!(reshape(left_tmp,l,p*r),term.factors[2],reshape(x_perm,l,p*r))
                    permutedims!(left_out,left_tmp,(2,1,3))
                end
                source = left_out
            end
            if !ri
                mul!(right_mat,reshape(source,p*l,r),Rt)
                source = right_out
            end
            if !bi
                mul!(acted_mat,term.factors[1],reshape(source,p,l*r))
                source = acted
            end
            dest .+= term.coefficient .* source
        end
        return dest
    end
end

# `chunks` > 1 splits the terms round-robin into groups with private scratch
# and partial sums, added in a fixed order: results do not depend on
# `threaded` or on the thread count. `threaded=true` runs groups as tasks.
function _core_terms_workspace(terms, initial; buffers=nothing, chunks=1, threaded=false, lowrank=false)
    p,l,r = size(initial)
    chunks = clamp(chunks,1,max(1,length(terms)))
    groups = [terms[i:chunks:end] for i in 1:chunks]
    if isnothing(buffers)
        first_scratch = (similar(initial),Array{ComplexF64}(undef,l,p,r),similar(initial),similar(initial))
        x_perm = Array{ComplexF64}(undef,l,p,r)
        extra = [_core_group_scratch(p,l,r) for _ in 2:chunks]
    else
        # Historical six-slot layout (all slots hold p*l*r elements) plus an
        # optional growable list of per-group scratch sets in slot seven.
        first_scratch = (reshape(buffers[1],p,l,r),reshape(buffers[3],l,p,r),buffers[5],buffers[6])
        x_perm = reshape(buffers[2],l,p,r)
        pool = length(buffers) >= 7 ? buffers[7] : Any[]
        while length(pool) < chunks-1
            push!(pool,_core_group_scratch(p,l,r))
        end
        extra = pool[1:chunks-1]
    end
    kernels = [_core_terms_kernel(groups[1],p,l,r,first_scratch;lowrank);
               [_core_terms_kernel(groups[k],p,l,r,extra[k-1][1:4];lowrank) for k in 2:chunks]]
    partials = [extra[k-1][5] for k in 2:chunks]
    needs_perm = !_slice_mode2(p,l) && any(t->!isone(t.factors[2]),terms)
    function action(x)
        result = zeros(ComplexF64,size(x))
        needs_perm && permutedims!(x_perm,x,(2,1,3))
        if chunks > 1 && threaded
            tasks = [Threads.@spawn (fill!(partials[k-1],0); kernels[k](partials[k-1],x,x_perm)) for k in 2:chunks]
            kernels[1](result,x,x_perm)
            foreach(fetch,tasks)
        else
            kernels[1](result,x,x_perm)
            for k in 2:chunks
                fill!(partials[k-1],0)
                kernels[k](partials[k-1],x,x_perm)
            end
        end
        for k in 2:chunks
            result .+= partials[k-1]
        end
        return result
    end
    return action
end
