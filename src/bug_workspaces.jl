# Scratch storage belongs to one sequential local Krylov solve. Never return
# a scratch array: Arnoldi retains action outputs while applying the operator
# again. Preserve the contraction and term accumulation order of _*_rhs.
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
    p,l,r = size(initial)
    left_input = Array{ComplexF64}(undef,l,p,r)
    left_output = similar(left_input)
    projected_left = similar(initial)
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
        for (k,term) in enumerate(operator.terms)
            iszero(term.coefficient) && continue
            permutedims!(left_input,x,(2,1,3))
            mul!(left_out,left[k],left_in)
            permutedims!(projected_left,left_output,(2,1,3))
            permutedims!(right_input,projected_left,(3,1,2))
            mul!(right_out,right[k],right_in)
            permutedims!(projected_right,right_output,(2,3,1))
            mul!(parent_out,boundary[k],parent_in)
            result .+= term.coefficient .* acted
        end
        return result
    end
    return action
end
