@testset "Local residual enrichment with certified acceptance" begin
    tree = balanced_tree(3)
    x = product_state(tree,[[1,0] for _ in 1:3])
    sx = ComplexF64[0 1;1 0]
    op = SumOfProducts([2,2,2],[ProductTerm(-0.7im,Dict(1=>sx,2=>sx)),
                              ProductTerm(-0.3im,Dict(2=>sx,3=>sx))])
    for scheme in (:cheb,:cn)
        local_y,local_d = tamen_step(x,op,0.4;time_scheme=scheme,time_points=5,
            rtol=1e-9,maxrank=8,residual_method=:local)
        global_y,global_d = tamen_step(x,op,0.4;time_scheme=scheme,time_points=5,
            rtol=1e-9,maxrank=8,residual_method=:global)
        @test ttn_distance(local_y,global_y) < 1e-9
        @test local_d.converged && local_d.history[end].global_residual_checked
        @test global_d.global_checks == global_d.sweeps
        @test local_d.global_checks < local_d.sweeps
        @test local_d.residual_updates > 0
        @test global_d.residual_updates == 0
        @test local_d.timings.residual_seconds ≈ local_d.timings.residual_local_seconds+local_d.timings.residual_global_seconds
        @test all(h->h.global_residual_checked || isinf(h.residual_bound),local_d.history)
        grid = TTNSolver._tamen_grid(5,0.4,scheme)
        A = TTNSolver._spacetime_operator(op,grid)
        b = TTNSolver._time_lift(x,grid.d0)
        @test norm(vec(dense(b))-dense(A)*vec(dense(local_d.solution.state))) <= local_d.residual_bound+1e-10
    end
    # Constrained solution and auxiliary ranks cannot silently turn a small
    # projected residual into a convergence certificate; final check is mandatory.
    _,bad = tamen_step(x,op,0.4;maxrank=1,kickrank=1,maxsweeps=3,
        rtol=1e-10,strict=false,residual_method=:local,residual_check_every=2)
    @test !bad.converged
    @test bad.history[2].global_residual_checked
    @test bad.history[end].global_residual_checked
    @test bad.residual_bound == bad.history[end].residual_bound
    @test bad.relative_residual_bound > 1e-10
    @test_throws TAMEnConvergenceError tamen_step(x,op,0.4;maxrank=1,kickrank=1,
        maxsweeps=2,rtol=1e-10,residual_method=:local)
    for kwargs in ((;residual_method=:unknown),(;residual_check_every=0),(;residual_sweeps=0))
        @test_throws ArgumentError tamen_step(x,op,0.1;kwargs...)
    end
end
