# Unprotected weighted cuts of a privately owned, root-orthogonal residual.
# The general/protected truncate! remains in truncation.jl. Cuts stay sequential
# in the same parent/child order, preserving its complementary-weight bound.
function _cut_factor!(work,node,rows,cols)
    data=work.factors[node]
    count=rows*cols
    if length(data)<count
        resize!(data,count);work.grows+=1
    else
        work.reuses+=1
    end
    return reshape(view(data,1:count),rows,cols)
end

function _cut_weight!(work,C,F,mode)
    p,l,r=size(C); f=size(F,1)
    amplitude=_residual_buffer!(work,1,f,l,r)
    mul!(reshape(amplitude,f,:),F,reshape(C,p,:))
    rows,cols=mode==2 ? (f*r,l) : (f*l,r)
    A=_residual_buffer!(work,2,rows,cols)
    if mode==2
        permutedims!(reshape(A,f,r,l),amplitude,(1,3,2))
    else
        copyto!(A,reshape(amplitude,rows,cols))
    end
    k=min(rows,cols)
    T=_residual_buffer!(work,3,min(36,k),k)
    LinearAlgebra.LAPACK.geqrt!(A,T)
    R=_residual_buffer!(work,4,k,cols)
    fill!(R,0)
    for j in 1:cols, i in 1:min(j,k)
        R[i,j]=A[i,j]
    end
    M=_residual_buffer!(work,5,cols,k)
    permutedims!(M,R,(2,1))
    return R,M
end

function _cut_space!(M,budget,maxrank)
    # Nothing denotes the identity cut. Keep R for descendants but avoid
    # identity products and avoid aliasing scratch into the returned tensor.
    iszero(budget) && maxrank>=size(M,1) && return nothing,0.0
    fac=svd!(M)
    count=0
    while count<length(fac.S) && norm(view(fac.S,count+1:length(fac.S)))>budget
        count+=1
    end
    count=min(max(count,1),maxrank,length(fac.S))
    return view(fac.U,:,1:count),norm(view(fac.S,count+1:length(fac.S)))
end

function _cut_child_product!(work,C,W,mode)
    p,l,r=size(C)
    n,s=mode==2 ? (l,r) : (r,l)
    input=_residual_buffer!(work,7,n,p,s)
    permutedims!(input,C,mode==2 ? (2,1,3) : (3,1,2))
    k=size(W,2)
    output=_residual_buffer!(work,8,k,p,s)
    mul!(reshape(output,k,:),W',reshape(input,n,:))
    return permutedims(output,mode==2 ? (2,1,3) : (2,3,1))
end

function _residual_truncate!(x,work::_ResidualCutWorkspace;atol,maxrank,timings=nothing)
    tree=x.tree; n=length(tree.children)
    while length(work.factors)<n
        push!(work.factors,ComplexF64[])
    end
    resize!(work.factors,n)
    factors=Vector{AbstractMatrix{ComplexF64}}(undef,n)
    factors[tree.root]=_cut_factor!(work,tree.root,1,1)
    factors[tree.root][1]=1
    budget=Float64(atol)/max(1,n-1)
    discarded=0.0
    for node in reverse(tree.postorder)
        _isleaf(tree,node) && continue
        for (child,mode) in zip(tree.children[node],(2,3))
            C=x.cores[node]
            started=_residual_tick(timings)
            R,M=_cut_weight!(work,C,factors[node],mode)
            _residual_record!(timings,Val(:weight_qr_seconds),started)
            started=_residual_tick(timings)
            W,drop=_cut_space!(M,budget,maxrank)
            discarded+=drop
            _residual_record!(timings,Val(:svd_seconds),started)
            started=_residual_tick(timings)
            rank=isnothing(W) ? size(R,2) : size(W,2)
            factors[child]=_cut_factor!(work,child,size(R,1),rank)
            if isnothing(W)
                copyto!(factors[child],R)
            else
                conjugate=_residual_buffer!(work,6,size(W)...)
                conjugate.=conj.(W)
                mul!(factors[child],R,conjugate)
                x.cores[node]=_cut_child_product!(work,C,W,mode)
                if _isleaf(tree,child)
                    x.leaves[child]=x.leaves[child]*W
                else
                    x.cores[child]=_parent_product(x.cores[child],transpose(W))
                end
            end
            _residual_record!(timings,Val(:basis_update_seconds),started)
        end
    end
    return (;ranks=tree_ranks(x),discarded_bound=discarded,tolerance_met=discarded<=Float64(atol))
end
