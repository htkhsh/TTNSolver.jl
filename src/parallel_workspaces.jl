# One rectangular SoP contraction with fixed input x. The scratch arrays are
# reused across terms, not across nodes or tasks. Output and scratch arrays use
# compact dimensions; no old-basis zero padding enters the matrix products.
function _parallel_core_block(operator,boundary,left,right,x,output_size)
    result = zeros(ComplexF64,output_size)
    (isempty(result) || isempty(operator.terms)) && return result
    terms = _group_core_terms(operator,boundary,left,right)
    p,l,r = size(x)
    a,b,c = output_size
    left_in = reshape(permutedims(x,(2,1,3)),l,:)
    left_out = Array{ComplexF64}(undef,b,p,r)
    right_in = Array{ComplexF64}(undef,r,p,b)
    right_out = Array{ComplexF64}(undef,c,p,b)
    parent_in = Array{ComplexF64}(undef,p,b,c)
    acted = Array{ComplexF64}(undef,a,b,c)
    lm,ri,ro = reshape(left_out,b,:),reshape(right_in,r,:),reshape(right_out,c,:)
    pi,po = reshape(parent_in,p,:),reshape(acted,a,:)
    for term in terms
        B,L,R = term.factors
        mul!(lm,L,left_in)
        # Directly shuffle left/parent/right -> right/parent/left.
        permutedims!(right_in,left_out,(3,2,1))
        mul!(ro,R,ri)
        permutedims!(parent_in,right_out,(2,3,1))
        mul!(po,B,pi)
        result .+= term.coefficient .* acted
    end
    return result
end
