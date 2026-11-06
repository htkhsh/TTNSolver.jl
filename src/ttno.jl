"""
    TreeTensorOperator(operator::SumOfProducts, tree; atol=0, maxrank=typemax(Int))

Compile a sum of products into a tree tensor network operator (TTNO). Identical
subtree products share a channel. Physical operator axes are vectorized with
the output index fastest. No global dense operator is formed.

Optional weighted compression reports a Frobenius error bound in
`discarded_bound` (roundoff excluded). With the defaults, compilation only
merges exact structural duplicates. Numeric arrays are owned and read-only;
the topology is shared. This experimental backend does not replace SoP BUG
contractions. Operator/state bond ranks multiply during `apply`.
"""
struct TreeTensorOperator
    tensor::TTNTensor
    dimensions::Vector{Int}
    discarded_bound::Float64
end

function TreeTensorOperator(operator::SumOfProducts,tree::BinaryTree;
                            atol=0.0,maxrank=typemax(Int))
    _compression_options(atol,maxrank)
    dims=leafdims(operator)
    length(dims)==length(tree.leaf_order) || throw(DimensionMismatch("operator and tree leaf counts differ"))
    terms=filter(t->!iszero(t.coefficient),operator.terms)
    if isempty(terms)
        vectors=[ones(ComplexF64,n*n) for n in dims]
        fill!(first(vectors),0)
        return TreeTensorOperator(product_state(tree,vectors),dims,0.0)
    end
    leaves=Dict{Int,Matrix{ComplexF64}}()
    cores=Dict{Int,Array{ComplexF64,3}}()
    channels=Vector{Vector{Int}}(undef,length(tree.children))
    ranks=zeros(Int,length(tree.children))
    for (mode,node) in enumerate(tree.leaf_order)
        n=dims[mode]
        identity=Matrix{ComplexF64}(I,n,n)
        if node==tree.root
            M=zeros(ComplexF64,n,n)
            for t in terms; M .+= t.coefficient .* get(t.factors,mode,identity); end
            leaves[node]=reshape(M,:,1)
            continue
        end
        lookup=Dict{Matrix{ComplexF64},Int}()
        factors=Matrix{ComplexF64}[]
        channels[node]=[get!(lookup,get(t.factors,mode,identity)) do
            push!(factors,get(t.factors,mode,identity))
            length(factors)
        end for t in terms]
        leaves[node]=hcat(vec.(factors)...)
        ranks[node]=length(factors)
    end
    for node in tree.postorder
        _isleaf(tree,node) && continue
        left,right=tree.children[node]
        pairs=collect(zip(channels[left],channels[right]))
        if node==tree.root
            C=zeros(ComplexF64,1,ranks[left],ranks[right])
            for (t,(l,r)) in zip(terms,pairs); C[1,l,r]+=t.coefficient; end
        else
            lookup=Dict{Tuple{Int,Int},Int}()
            channels[node]=[get!(lookup,pair,length(lookup)+1) for pair in pairs]
            ranks[node]=length(lookup)
            C=zeros(ComplexF64,ranks[node],ranks[left],ranks[right])
            for ((l,r),p) in lookup; C[p,l,r]=1; end
        end
        cores[node]=C
    end
    tensor=TTNTensor(tree,leaves,cores)
    bound=0.0
    if atol>0 || maximum(tree_ranks(tensor))>maxrank
        bound=truncate!(tensor;atol,maxrank).discarded_bound
    end
    return TreeTensorOperator(tensor,dims,bound)
end

leafdims(A::TreeTensorOperator)=copy(A.dimensions)
tree_ranks(A::TreeTensorOperator)=tree_ranks(A.tensor)

"""Dense matrix of the represented TTNO, for small reference problems only."""
function dense(A::TreeTensorOperator)
    d=length(A.dimensions)
    shape=Tuple(n for n in A.dimensions for _ in 1:2)
    interleaved=reshape(dense(A.tensor),shape)
    return reshape(permutedims(interleaved,vcat(collect(1:2:2d),collect(2:2:2d))),
                   prod(A.dimensions),prod(A.dimensions))
end

function _check_operator(A::TreeTensorOperator,x::TTNTensor)
    leafdims(x)==A.dimensions || throw(DimensionMismatch("operator and state physical dimensions differ"))
    a,b=A.tensor.tree,x.tree
    a.root==b.root && a.children==b.children && a.leaf_order==b.leaf_order ||
        throw(ArgumentError("operator and state trees differ"))
end

"""
    apply(operator::TreeTensorOperator, state::TTNTensor; max_intermediate_bytes=2^30)

Apply the represented TTNO and QR each result subtree once. Does not truncate
the state; returns independent, root-orthogonal storage. Nonzero transfer
entries accumulate into shared channels, avoiding termwise TTN sums. Predicts
the largest local QR input before allocation and rejects it above the specified
limit (default 1 GiB). This is a per-array limit, not a bound on peak memory.
"""
function apply(A::TreeTensorOperator,x::TTNTensor;max_intermediate_bytes=2^30)
    _check_operator(A,x)
    max_intermediate_bytes isa Integer && max_intermediate_bytes>0 ||
        throw(ArgumentError("max_intermediate_bytes must be a positive integer"))
    required=_ttno_application_bytes(A,x)
    required<=max_intermediate_bytes || throw(ArgumentError(
        "TTNO application needs a local QR input of $required bytes (limit $max_intermediate_bytes); " *
        "use the SoP residual backend or explicitly increase apply's max_intermediate_bytes"))
    tree=x.tree
    leaves=Dict{Int,Matrix{ComplexF64}}()
    cores=Dict{Int,Array{ComplexF64,3}}()
    coordinates=Vector{Array{ComplexF64,3}}(undef,length(tree.children))
    modes=Dict(node=>mode for (mode,node) in enumerate(tree.leaf_order))
    for node in tree.postorder
        leaf=_isleaf(tree,node)
        if leaf
            n=A.dimensions[modes[node]]
            U=x.leaves[node]; W=A.tensor.leaves[node]
            px,wp=size(U,2),size(W,2)
            M=Matrix{ComplexF64}(undef,n,px*wp)
            for a in 1:wp
                mul!(view(M,:,(a-1)*px+1:a*px),reshape(view(W,:,a),n,n),U)
            end
        else
            left,right=tree.children[node]
            L,R=coordinates[left],coordinates[right]
            C,W=x.cores[node],A.tensor.cores[node]
            dl,dr=size(L,1),size(R,1)
            px,wp=size(C,1),size(W,1)
            M=zeros(ComplexF64,dl*dr,px*wp)
            # Project each distinct child-channel pair once; share its result
            # across all parent channels. No zero-filled block-sum tensor.
            for c in axes(W,3), b in axes(W,2)
                all(iszero,view(W,:,b,c)) && continue
                projected=_basis_matrix(_project_children(C,view(L,:,:,b),view(R,:,:,c)))
                for a in axes(W,1)
                    value=W[a,b,c]; iszero(value) && continue
                    view(M,:,(a-1)*px+1:a*px) .+= value .* projected
                end
            end
        end
        if node==tree.root
            if leaf; leaves[node]=M
            else; cores[node]=reshape(M,1,dl,dr)
            end
            continue
        end
        F=qr(M)
        rank=min(size(M)...)
        Q=Matrix(F.Q)[:,1:rank]
        coordinates[node]=reshape(Matrix(F.R),rank,px,wp)
        if leaf; leaves[node]=Q
        else; cores[node]=permutedims(reshape(Q,dl,dr,rank),(3,1,2))
        end
    end
    return TTNTensor(tree,leaves,cores)
end

Base.:*(A::TreeTensorOperator,x::TTNTensor)=apply(A,x)

# Exact QR shape prediction. BigInt prevents integer wraparound from bypassing
# the allocation guard for large rank products. No numerical array is created.
function _ttno_application_bytes(A::TreeTensorOperator,x::TTNTensor)
    ranks=zeros(BigInt,length(x.tree.children))
    largest=big(0)
    for node in x.tree.postorder
        leaf=_isleaf(x.tree,node)
        rows=if leaf
            big(size(x.leaves[node],1))
        else
            left,right=x.tree.children[node];ranks[left]*ranks[right]
        end
        wp=leaf ? size(A.tensor.leaves[node],2) : size(A.tensor.cores[node],1)
        xp=leaf ? size(x.leaves[node],2) : size(x.cores[node],1)
        columns=big(wp)*xp
        largest=max(largest,rows*columns)
        ranks[node]=min(rows,columns)
    end
    return sizeof(ComplexF64)*largest
end
