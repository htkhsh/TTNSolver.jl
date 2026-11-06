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

function _core_terms_workspace(terms, initial; buffers=nothing)
    identities = [isone.(t.factors) for t in terms]
    transposed = [(transpose(t.factors[2]),transpose(t.factors[3])) for t in terms]
    p,l,r = size(initial)
    if isnothing(buffers)
        left_out,right_out,acted = similar(initial),similar(initial),similar(initial)
        x_perm,left_tmp = Array{ComplexF64}(undef,l,p,r),Array{ComplexF64}(undef,l,p,r)
    else
        # Historical six-slot layout; all slots hold p*l*r elements.
        left_out = reshape(buffers[1],p,l,r)
        x_perm,left_tmp = reshape(buffers[2],l,p,r),reshape(buffers[3],l,p,r)
        right_out,acted = buffers[5],buffers[6]
    end
    right_mat,acted_mat = reshape(right_out,p*l,r),reshape(acted,p,l*r)
    slices = _slice_mode2(p,l)
    has_left = any(ids->!ids[2],identities)
    # Child factors act on x directly: the right factor on a (p*l,r) view, the
    # boundary on (p,l*r), and the left factor per right-index slice (or, for
    # small p, on x permuted once per action).
    function action(x)
        result = zeros(ComplexF64,size(x))
        !slices && has_left && permutedims!(x_perm,x,(2,1,3))
        for (term,(bi,li,ri),(Lt,Rt)) in zip(terms,identities,transposed)
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
            result .+= term.coefficient .* source
        end
        return result
    end
    return action
end
