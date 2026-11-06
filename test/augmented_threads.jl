@testset "Augmented BUG subtree threading" begin
    rng = MersenneTwister(417)
    trees = (balanced_tree(1), balanced_tree(5),
        BinaryTree([(2,5),(0,0),(0,0),(0,0),(4,3)];root=1,leaf_order=[3,2,4]))
    for tree in trees
        n = length(tree.leaf_order)
        x = product_state(tree,[randn(rng,ComplexF64,3) for _ in 1:n])
        c = product_state(tree,[ones(ComplexF64,3) for _ in 1:n])
        A = ComplexF64[-1 0.2im 0.3;0.4 -0.2im -0.1;0.6 0 -0.2]
        op = SumOfProducts(fill(3,n),[ProductTerm(0.2,Dict(j=>A)) for j in 1:n])
        before = dense(x)
        for invariant in (nothing,c), correct in (false,true)
            opts = (;atol=1e-12,maxrank=16,conserved=invariant,correct_invariant=correct)
            a, da = augmented_bug_step(x,op,0.04;opts...,threaded=false)
            b, db = augmented_bug_step(x,op,0.04;opts...,threaded=true)
            @test ttn_distance(a,b) < 1e-12*norm(x)
            @test da.ranks == db.ranks
            @test da.local_solves == db.local_solves
            @test da.skipped_leaf_solves == db.skipped_leaf_solves
            @test da.krylov_error_estimate == db.krylov_error_estimate
            @test db.threaded == (Threads.nthreads(:default) > 1 && n > 1)
            @test !da.threaded && da.method == db.method == :augmented
            # Reduced bases retain time-discretization error even for this
            # separable generator; exact full-rank checks live in bug.jl.
            @test norm(vec(dense(b))-exp(0.04*dense(op))*vec(before)) < 1e-5*norm(x)
            isnothing(invariant) || @test dot(c,b) ≈ dot(c,x) atol=1e-11
        end
        @test dense(x) == before
    end

    # Repeated steps exercise internal enlarged bases and full leaves; reuse
    # one preparation concurrently to catch workspace or output aliasing.
    tree = balanced_tree(5)
    x = product_state(tree,[ComplexF64[1,0] for _ in 1:5])
    A = ComplexF64[0 1;1 0]
    op = SumOfProducts(fill(2,5),[ProductTerm(-0.2im,Dict(j=>A,j+1=>A)) for j in 1:4])
    for _ in 1:3
        a,da = augmented_bug_step(x,op,0.03;threaded=false)
        b,db = augmented_bug_step(x,op,0.03;threaded=true)
        @test ttn_distance(a,b) < 1e-13
        @test da.ranks == db.ranks
        x = a
    end
    prepared = TTNSolver._prepare_bug_step(x,op,nothing,32)
    before = dense(prepared.old)
    tasks = [Threads.@spawn TTNSolver._prepared_bug_step(prepared,op,0.03;threaded=true) for _ in 1:3]
    results = fetch.(tasks)
    @test all(ttn_distance(first(results)[1],r[1]) < 1e-13 for r in results)
    first(results)[1].cores[tree.root] .= 0
    @test dense(prepared.old) == before
    @test norm(last(results)[1]) > 0.9
    zero = copy(x)
    zero.cores[tree.root] .= 0
    z,dz = augmented_bug_step(zero,op,0.01;threaded=true)
    @test norm(z) == 0

    # Failed child tasks must finish before returning the original exception;
    # adaptive step rejection must still recognize a local Arnoldi failure.
    A4 = ComplexF64[-1 1 0 0;0 -2 1 0;0 0 -3 1;1 0 0 -4]
    x = product_state(balanced_tree(3),[[1,0,0,0] for _ in 1:3])
    op = SumOfProducts(fill(4,3),[ProductTerm(1,Dict(j=>A4)) for j in 1:3])
    @test_throws LocalSolveFailure augmented_bug_step(x,op,0.05;threaded=true,
        krylovdim=2,krylov_maxiter=1,krylov_tol=0.01)
    serial = adaptive_evolve(x,op,(0.,0.05);dt=0.05,threaded=false,
        krylovdim=2,krylov_maxiter=1,krylov_tol=0.01,atol=1e-3,rtol=0,maxsteps=100)
    threaded = adaptive_evolve(x,op,(0.,0.05);dt=0.05,threaded=true,
        krylovdim=2,krylov_maxiter=1,krylov_tol=0.01,atol=1e-3,rtol=0,maxsteps=100)
    @test any(d -> d.reason == :local_solve,threaded.diagnostics)
    @test last(threaded.diagnostics).accepted
    @test serial.stats == threaded.stats
    @test ttn_distance(serial.final_state,threaded.final_state) < 1e-13

    # Concurrent failures must preserve programming errors, and all work must
    # have stopped when the caller catches the exception.
    completed = zeros(Int,2)
    update = child -> begin
        yield()
        completed[child] = 1
        child == 1 ? throw(LocalSolveFailure(:nonconvergence,1.)) : throw(ArgumentError("test error"))
    end
    @test_throws ArgumentError TTNSolver._bug_subtrees!(update,1,2,true)
    @test completed == [1,1]
end
