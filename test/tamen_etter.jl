function etter_test_state(tree,dims,rank,rng)
    leaves = Dict{Int,Matrix{ComplexF64}}()
    cores = Dict{Int,Array{ComplexF64,3}}()
    ranks = zeros(Int,length(tree.children))
    for (node,d) in zip(tree.leaf_order,dims)
        r = min(d,rank)
        leaves[node] = Matrix(qr(randn(rng,ComplexF64,d,r)).Q)[:,1:r]
        ranks[node] = r
    end
    for node in tree.postorder
        haskey(leaves,node) && continue
        a,b = tree.children[node]
        r = node == tree.root ? 1 : min(rank,ranks[a]*ranks[b])
        Q = Matrix(qr(randn(rng,ComplexF64,ranks[a]*ranks[b],r)).Q)[:,1:r]
        cores[node] = reshape(Matrix(transpose(Q)),r,ranks[a],ranks[b])
        ranks[node] = r
    end
    return TTNTensor(tree,leaves,cores)
end

@testset "Etter partial RHS frames" begin
    rng = MersenneTwister(882)
    C = randn(rng,ComplexF64,2,5,3) # shrinking left bond: 5 -> 5, right: 3 -> 3
    B = [randn(rng,ComplexF64,2,2)]
    L,R = [randn(rng,ComplexF64,5,5)],[randn(rng,ComplexF64,3,3)]
    rhs = randn(rng,ComplexF64,2,2,4)
    cl,cr = randn(rng,ComplexF64,5,2),randn(rng,ComplexF64,3,4)
    p = TTNSolver._etter_child_frames(C,B,L,R,rhs;cross_left=cl,cross_right=cr)
    F = svd(TTNSolver._complement_matrix(C,2))
    @test p.left.rhs ≈ F.U'*kron(cr,Matrix{ComplexF64}(I,2,2))*TTNSolver._complement_matrix(rhs,2)
    G = svd(TTNSolver._complement_matrix(TTNSolver._child_product(C,conj(F.Vt),2),3))
    @test p.right.rhs ≈ G.U'*kron(conj(F.Vt)*cl,Matrix{ComplexF64}(I,2,2))*TTNSolver._complement_matrix(rhs,3)
    # A locally overcomplete bond shrinks without dropping a represented vector.
    C = randn(rng,ComplexF64,1,6,2)
    p = TTNSolver._etter_child_frames(C,[ones(ComplexF64,1,1)],
        [Matrix{ComplexF64}(I,6,6)],[Matrix{ComplexF64}(I,2,2)],C)
    @test size(p.left.transfer)==(2,6)
    @test p.left.operator[1] ≈ I
end

@testset "Etter worker ownership and failure" begin
    rng = MersenneTwister(884)
    tree = balanced_tree(8)
    initial = etter_test_state(tree,fill(4,8),2,rng)
    target = etter_test_state(tree,fill(4,8),2,rng)
    A = SumOfProducts(fill(4,8),[ProductTerm(1,Dict())])
    serial,parallel = copy(initial),copy(initial)
    options = (;tol=1e-11,krylovdim=40,krylov_maxiter=100)
    w = TAMEnWorkspace()
    s = TTNSolver._tamen_etter_sweep!(serial,A,target,0;options...,workspace=TAMEnWorkspace())
    p = TTNSolver._tamen_etter_sweep!(parallel,A,target,0;options...,workspace=w,threaded=true)
    @test ttn_distance(serial,parallel)<1e-11
    @test s.matvecs==p.matvecs && s.local_solves==p.local_solves
    @test p.spawned_tasks==min(Threads.nthreads(:default),8)-1
    # No alias between scratch owned by different nodes; repeat to reuse slots.
    allbuffers = [buf for entry in values(w.buffers) for buf in entry.buffers]
    @test length(Set(objectid.(allbuffers)))==length(allbuffers)
    allocations = w.buffer_allocations
    TTNSolver._tamen_etter_sweep!(copy(initial),A,target,0;options...,workspace=w,threaded=true)
    @test w.buffer_allocations==allocations && w.buffer_reuses>0
    @test norm(initial)>0 # initial remains independent of worker output

    # Root scalar solve converges; both child GMRES solves fail. All started
    # children must join, preserve the original error type, and publish no tree.
    tree = balanced_tree(2)
    x = product_state(tree,[[1.,1,1,1],[1.,2,3,4]])
    before = dense(x)
    b = product_state(tree,[[1.,0,1,0],[0.,1,0,1]])
    D = diagm(ComplexF64[1,2,3,4])
    op = SumOfProducts([4,4],[ProductTerm(1,Dict(1=>D,2=>D))])
    w = TAMEnWorkspace()
    err = try
        TTNSolver._tamen_etter_sweep!(x,op,b,0;tol=1e-10,krylovdim=1,
            krylov_maxiter=1,workspace=w,threaded=true)
        nothing
    catch e
        e
    end
    @test err isa TAMEnConvergenceError
    @test err.reason==:local_gmres && err.diagnostics.node!=tree.root
    @test dense(x)==before && isempty(w.buffers)
    # A successful immediate reuse exercises that no worker outlives failure.
    TTNSolver._tamen_etter_sweep!(x,SumOfProducts([4,4],[ProductTerm(1,Dict())]),b,0;
        options...,workspace=w,threaded=true)
    @test dense(x) ≈ dense(b) atol=1e-10
end

@testset "Public Etter tAMEn integration" begin
    sx = ComplexF64[0 1;1 0]
    tree = BinaryTree([(2,3),(0,0),(4,5),(0,0),(0,0)];root=1,leaf_order=[5,2,4])
    x = product_state(tree,[[1,0],[1,0],[1,0]])
    L = SumOfProducts([2,2,2],[ProductTerm(-0.7im,Dict(1=>sx,2=>sx)),
        ProductTerm(-0.3im,Dict(2=>sx,3=>sx))])
    options = (;time_points=7,rtol=1e-9,maxrank=8,maxsweeps=12)
    serial,si = tamen_step(x,L,0.4;options...,als_sweep=:etter)
    y,d = tamen_step(x,L,0.4;options...,als_sweep=:etter,threaded=true,store_warm_start=true)
    @test d.als_sweep==:etter && si.als_parallel_tasks==0
    @test (d.als_parallel_tasks>0)==(Threads.nthreads(:default)>1)
    @test d.sweeps==si.sweeps && d.matvecs==si.matvecs
    @test ttn_distance(y,serial)<1e-11
    @test norm(vec(dense(y))-exp(0.4dense(L))*vec(dense(x)))<1e-8
    @test d.relative_residual_bound<=1e-9 && d.history[end].global_residual_checked
    @test 0<d.timings.sweep_seconds<=d.timings.als_seconds
    @test d.solution.state.leaves[d.solution.time_leaf] ≈ Matrix{ComplexF64}(I,7,7)
    yn,dn = tamen_step(y,L,0.4;options...,als_sweep=:etter,threaded=true,warm_start=d.warm_start)
    @test dn.warm_start_used
    @test norm(vec(dense(yn))-exp(0.8dense(L))*vec(dense(x)))<3e-8
    @test_throws ArgumentError tamen_step(x,L,0.4;als_sweep=:unknown)

    zeroop = SumOfProducts([2,2,2],ProductTerm[])
    z = product_state(tree,[[0,0],[1,0],[1,0]])
    yz,dz = tamen_step(z,zeroop,0.4;als_sweep=:etter,threaded=true)
    @test norm(yz)==0 && dz.converged
    # Multiple invariants must survive worker basis updates and final truncation.
    stationary = SumOfProducts([2,2],[ProductTerm(1,Dict(2=>ComplexF64[-1 2;1 -2]))])
    initial = product_state(balanced_tree(2),[[1,2],[2,0]])
    c = product_state(initial.tree,[[1,0],[1,1]])
    c2 = product_state(initial.tree,[[0,1],[1,1]])
    for scheme in (:cheb,:cn)
        result = tamen_evolve(initial,stationary,(0.,0.2);dt=0.1,dtmax=0.1,
            time_points=5,time_scheme=scheme,rtol=1e-4,solve_rtol=1e-10,
            als_sweep=:etter,threaded=true,conserved_observables=[c,c2])
        @test dot(c,result.final_state) ≈ 2 atol=1e-10
        @test dot(c2,result.final_state) ≈ 4 atol=1e-10
        @test all(a->a.solve.als_sweep==:etter,result.diagnostics)
    end
end

@testset "Etter frozen child frames" begin
    rng = MersenneTwister(729)
    # Each cached complement must be the orthonormal frame used to define
    # that child's local problem, including complex gauge transformations.
    C = randn(rng,ComplexF64,2,3,4)
    B = [randn(rng,ComplexF64,2,2)]
    L = [randn(rng,ComplexF64,3,3)]
    R = [randn(rng,ComplexF64,4,4)]
    rhs = randn(rng,ComplexF64,2,3,4)
    prepared = TTNSolver._etter_child_frames(C,B,L,R,rhs)
    # Independently construct the two sequential SVD frames from Definition33.
    F = svd(TTNSolver._complement_matrix(C,2))
    Q = F.U
    @test prepared.left.operator[1] ≈ Q'*kron(R[1],B[1])*Q atol=1e-11
    @test prepared.left.rhs ≈ Q'*TTNSolver._complement_matrix(rhs,2) atol=1e-11
    @test prepared.left.transfer ≈ Diagonal(F.S)*F.Vt atol=1e-11
    Crot = TTNSolver._child_product(C,conj(F.Vt),2)
    # C' in coefficient coordinates equals U*S (no singular-value inverse).
    G = svd(TTNSolver._complement_matrix(Crot,3))
    Lrot = conj(F.Vt)*L[1]*transpose(F.Vt)
    @test prepared.right.operator[1] ≈ G.U'*kron(Lrot,B[1])*G.U atol=1e-10
    rhsrot = TTNSolver._child_product(rhs,conj(F.Vt),2)
    @test prepared.right.rhs ≈ G.U'*TTNSolver._complement_matrix(rhsrot,3) atol=1e-10
    for data in (zeros(ComplexF64,2,3,4),C*1e-200)
        p = TTNSolver._etter_child_frames(data,B,L,R,rhs)
        @test all(isfinite,p.left.transfer) && all(isfinite,p.right.transfer)
        @test all(isfinite,p.left.operator[1]) && all(isfinite,p.right.operator[1])
    end
end

@testset "Serialized Etter solution sweep" begin
    rng = MersenneTwister(930)
    trees = (balanced_tree(4),BinaryTree([(2,3),(0,0),(4,5),(0,0),(0,0)];root=1,leaf_order=[5,2,4]))
    for tree in trees
        d = length(tree.leaf_order)
        x = etter_test_state(tree,fill(2,d),4,rng)
        # An exact fixed point for a non-Hermitian separable invertible operator.
        mats = [Matrix{ComplexF64}(I,2,2)+0.1randn(rng,ComplexF64,2,2) for _ in 1:d]
        A = SumOfProducts(fill(2,d),[ProductTerm(1,Dict(j=>mats[j] for j in 1:d))])
        b = A*x
        original = dense(x)
        s = TTNSolver._tamen_etter_sweep!(x,A,b,0;tol=1e-11,krylovdim=40,
            krylov_maxiter=100,workspace=TAMEnWorkspace(),threaded=false)
        @test dense(x) ≈ original atol=2e-10
        @test s.local_solves>0 && s.spawned_tasks==0
        for node in tree.postorder
            node==tree.root && continue
            U = TTNSolver._isleaf(tree,node) ? x.leaves[node] : TTNSolver._basis_matrix(x.cores[node])
            @test U'*U ≈ I atol=1e-11
        end
        # With complete bases, one sweep of identity-operator ALS recovers b.
        identity = SumOfProducts(fill(2,d),[ProductTerm(1,Dict())])
        target = etter_test_state(tree,fill(2,d),4,rng)
        TTNSolver._tamen_etter_sweep!(x,identity,target,0;tol=1e-11,krylovdim=40,
            krylov_maxiter=100,workspace=TAMEnWorkspace(),threaded=false)
        @test dense(x) ≈ dense(target) atol=2e-10
    end
end
