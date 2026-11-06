@testset "Parallel adaptive preparation and failure recovery" begin
    tree = balanced_tree(3)
    x = product_state(tree,[ComplexF64[1,0] for _ in 1:3])
    c = product_state(tree,[ones(ComplexF64,2) for _ in 1:3])
    A = ComplexF64[-1 0.3;1 -0.3]
    op = SumOfProducts(fill(2,3),[ProductTerm(1,Dict(1=>A,3=>A)),ProductTerm(0.2,Dict(2=>A))])
    options = (;conserved=c,maxrank=8,correct_invariant=false,
               krylov_tol=1e-12,krylovdim=30,krylov_maxiter=100)
    for (h1,h2) in ((0.04,0.04),(0.03,0.07)), threaded in (false,true)
        coarse,_ = parallel_bug_step(x,op,h1+h2;atol=1e-10,options...,threaded)
        middle,_ = parallel_bug_step(x,op,h1;atol=5e-11,options...,threaded)
        fine,_ = parallel_bug_step(middle,op,h2;atol=5e-11,options...,threaded)
        trial = TTNSolver._adaptive_trial(x,op,h1,h2,1e-10,options;method=:parallel,threaded)
        @test trial.reason == :candidate
        @test ttn_distance(trial.candidate,fine) < 1e-13
        @test trial.difference ≈ ttn_distance(coarse,fine) atol=1e-13
    end
    # A worker failure must arrive at the driver as LocalSolveFailure, so a
    # smaller step can recover rather than leaking TaskFailedException.
    A4 = ComplexF64[-1 1 0 0;0 -2 1 0;0 0 -3 1;1 0 0 -4]
    x = product_state(balanced_tree(3),[[1,0,0,0] for _ in 1:3])
    op = SumOfProducts(fill(4,3),[ProductTerm(1,Dict(j=>A4)) for j in 1:3])
    recovered = adaptive_evolve(x,op,(0.,0.05);dt=0.05,method=:parallel,
        krylovdim=2,krylov_maxiter=1,krylov_tol=0.01,atol=1e-3,rtol=0,maxsteps=100)
    @test any(d -> d.reason == :local_solve,recovered.diagnostics)
    @test last(recovered.diagnostics).accepted
    @test norm(vec(dense(recovered.final_state))-exp(0.05*dense(op))*vec(dense(x))) < 1e-3
    for invalid in (:other,"parallel",nothing,missing)
        @test_throws ArgumentError adaptive_evolve(x,op,(0.,0.);dt=0.1,method=invalid)
    end
end
