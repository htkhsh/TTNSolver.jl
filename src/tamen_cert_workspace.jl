# Each node owns eight high-water buffers: two-mode projection intermediates
# (1:4), joined QR input (5), block reflectors (6), thin Q (7), and R (8).
# R views are messages for this addition only; returned tensors never alias them.
Base.@kwdef mutable struct _ResidualTimings
    addition_seconds::Float64=0.0
    weight_qr_seconds::Float64=0.0
    svd_seconds::Float64=0.0
    basis_update_seconds::Float64=0.0
    final_qr_seconds::Float64=0.0
end
_residual_tick(::Nothing)=UInt64(0)
_residual_tick(::_ResidualTimings)=time_ns()
_residual_record!(::Nothing,::Val,started)=nothing
function _residual_record!(t::_ResidualTimings,::Val{field},started) where field
    setfield!(t,field,getfield(t,field)+(time_ns()-started)/1e9)
    return nothing
end
_residual_times(w)=NamedTuple{fieldnames(_ResidualTimings)}(
    ntuple(i->getfield(w.timings,i),fieldcount(_ResidualTimings)))
function _residual_time_delta(w,before)
    after=_residual_times(w)
    return NamedTuple{keys(after)}(values(after).-values(before))
end

mutable struct _ResidualNodeScratch
    storage::NTuple{8,Vector{ComplexF64}}
    grows::Int
    reuses::Int
    tasks::Int
end
_ResidualNodeScratch() = _ResidualNodeScratch(ntuple(_->ComplexF64[],8),0,0,0)

mutable struct _ResidualCutWorkspace
    storage::NTuple{8,Vector{ComplexF64}}
    factors::Vector{Vector{ComplexF64}}
    grows::Int
    reuses::Int
end
_ResidualCutWorkspace()=_ResidualCutWorkspace(ntuple(_->ComplexF64[],8),Vector{ComplexF64}[],0,0)

mutable struct _ResidualWorkspace
    busy::Threads.Atomic{Bool}
    nodes::Vector{_ResidualNodeScratch}
    grows::Int
    reuses::Int
    tasks::Int
    timings::_ResidualTimings
    cut::_ResidualCutWorkspace
end
_ResidualWorkspace() = _ResidualWorkspace(Threads.Atomic{Bool}(false),_ResidualNodeScratch[],0,0,0,
    _ResidualTimings(),_ResidualCutWorkspace())

function _with_residual_workspace(f,workspace)
    work = isnothing(workspace) ? _ResidualWorkspace() : workspace
    work isa _ResidualWorkspace || throw(ArgumentError("invalid residual workspace"))
    Threads.atomic_cas!(work.busy,false,true) && throw(ArgumentError("residual workspace is already in use"))
    try
        return f(work)
    finally
        Threads.atomic_xchg!(work.busy,false)
    end
end

_residual_work_counts(w) = (residual_buffer_grows=w.grows+w.cut.grows,
    residual_buffer_reuses=w.reuses+w.cut.reuses,residual_tasks=w.tasks)
# Retained numeric payload, excluding vector capacity overhead and object headers.
_residual_buffer_bytes(w) = sizeof(ComplexF64)*(sum(length(v) for node in w.nodes for v in node.storage;init=0)+
    sum(length,w.cut.storage;init=0)+sum(length,w.cut.factors;init=0))

function _prepare_residual_nodes!(work,n)
    while length(work.nodes)<n
        push!(work.nodes,_ResidualNodeScratch())
    end
    resize!(work.nodes,n)
    for node in work.nodes
        node.grows=node.reuses=node.tasks=0
    end
end

function _residual_buffer!(node,slot,dims...)
    data=node.storage[slot]
    count=prod(dims)
    if length(data)<count
        resize!(data,count)
        node.grows+=1
    else
        node.reuses+=1
    end
    return reshape(view(data,1:count),dims...)
end

# Fill a parent-column block directly, without two intermediate coefficient
# tensors and their unfolding copies. Coefficients use transpose, not adjoint.
function _residual_project_block!(dest,C,L,R,node)
    p,l,r=size(C)
    a,b=size(L,1),size(R,1)
    input=_residual_buffer!(node,1,l,p,r)
    left=_residual_buffer!(node,2,a,p,r)
    rightinput=_residual_buffer!(node,3,r,p,a)
    right=_residual_buffer!(node,4,b,p,a)
    permutedims!(input,C,(2,1,3))
    mul!(reshape(left,a,:),L,reshape(input,l,:))
    permutedims!(rightinput,left,(3,2,1))
    mul!(reshape(right,b,:),R,reshape(rightinput,r,:))
    permutedims!(reshape(dest,a,b,p),right,(3,1,2))
    return nothing
end

function _residual_qr!(node,M)
    m,n=size(M); k=min(m,n)
    T=_residual_buffer!(node,6,min(36,k),k)
    Q=_residual_buffer!(node,7,m,k)
    R=_residual_buffer!(node,8,k,n)
    # Same blocked factorization as qr!, but reuse T and materialize Q in place.
    LinearAlgebra.LAPACK.geqrt!(M,T)
    fill!(R,0)
    for j in 1:n, i in 1:min(j,k)
        R[i,j]=M[i,j]
    end
    fill!(Q,0)
    for j in 1:k
        Q[j,j]=1
    end
    LinearAlgebra.LAPACK.gemqrt!('L','N',M,T,Q)
    return Q,R
end
