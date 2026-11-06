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
function _leaf_rhs_workspace(operator, mode, boundary, initial)
    local_product = similar(initial)
    acted = similar(initial)
    transposed = transpose.(boundary)
    function action(x)
        result = zeros(ComplexF64,size(x))
        for (k,term) in enumerate(operator.terms)
            iszero(term.coefficient) && continue
            if haskey(term.factors,mode)
                mul!(local_product,term.factors[mode],x)
                mul!(acted,local_product,transposed[k])
            else
                mul!(acted,x,transposed[k])
            end
            result .+= term.coefficient .* acted
        end
        return result
    end
    return action
end

function _core_rhs_workspace(operator, boundary, left, right, initial)
    terms = _group_core_terms(operator, boundary, left, right)
    p,l,r = size(initial)
    left_input = Array{ComplexF64}(undef,l,p,r)
    left_output = similar(left_input)
    right_input = Array{ComplexF64}(undef,r,p,l)
    right_output = similar(right_input)
    projected_right = similar(initial)
    acted = similar(initial)
    # Cache matrix views too; their array storage is shared with the tensors.
    left_in,left_out = reshape(left_input,l,:),reshape(left_output,l,:)
    right_in,right_out = reshape(right_input,r,:),reshape(right_output,r,:)
    parent_in,parent_out = reshape(projected_right,p,:),reshape(acted,p,:)
    function action(x)
        result = zeros(ComplexF64,size(x))
        permutedims!(left_input,x,(2,1,3))
        for term in terms
            B,L,R = term.factors
            mul!(left_out,L,left_in)
            permutedims!(right_input,left_output,(3,2,1))
            mul!(right_out,R,right_in)
            permutedims!(projected_right,right_output,(2,3,1))
            mul!(parent_out,B,parent_in)
            result .+= term.coefficient .* acted
        end
        return result
    end
    return action
end
