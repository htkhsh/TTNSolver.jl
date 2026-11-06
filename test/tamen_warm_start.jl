@testset "Owned cross-interval tAMEn guesses" begin
    tree = balanced_tree(3)
    x = product_state(tree,[[1,0] for _ in 1:3])
    sx = ComplexF64[0 1;1 0]
    op = SumOfProducts([2,2,2],[ProductTerm(-0.7im,Dict(1=>sx,2=>sx)),
                              ProductTerm(-0.3im,Dict(2=>sx,3=>sx))])
    y,d = tamen_step(x,op,0.2;time_points=5,rtol=1e-10,maxrank=8,store_warm_start=true)
    @test d.warm_start isa TAMEnWarmStart
    @test !d.warm_start_used
    hint = d.warm_start
    stored = dense(hint.solution.state)
    @test hint.solution.state !== d.solution.state
    @test hint.residual !== nothing
    for (h,nt,scheme) in ((0.2,5,:cheb),(0.1,5,:cheb),(0.3,7,:cheb),(0.1,5,:cn))
        cold,cd = tamen_step(y,op,h;time_points=nt,time_scheme=scheme,rtol=1e-10,maxrank=8)
        warm,wd = tamen_step(y,op,h;time_points=nt,time_scheme=scheme,rtol=1e-10,
            maxrank=8,warm_start=hint,store_warm_start=true)
        @test wd.warm_start_used && wd.warm_start_status == :used
        @test wd.residual_warm_start_used == (nt==5 && scheme==:cheb)
        @test ttn_distance(warm,cold) < 1e-9
        @test wd.history[end].global_residual_checked && wd.relative_residual_bound <= 1e-10
        @test dense(time_slice(wd.solution,1)) == dense(y)
        @test wd.solution.state.leaves[wd.solution.time_leaf] ≈ Matrix{ComplexF64}(I,nt,nt)
        @test wd.warm_start !== hint
        @test cd.warm_start === nothing
    end
    @test dense(hint.solution.state) == stored
    # The solver RHS uses the NEW generator; no projected operator is cached.
    changed = SumOfProducts([2,2,2],[ProductTerm(0.2im,Dict(1=>sx))])
    v,vd = tamen_step(y,changed,0.1;warm_start=hint,rtol=1e-10,maxrank=8)
    @test norm(vec(dense(v))-exp(0.1dense(changed))*vec(dense(y))) < 1e-9
    @test vd.warm_start_used
    # A hint from a different trajectory/layout is discarded, never applied as
    # the new initial condition.
    _,wrong = tamen_step(x,op,0.1;warm_start=hint)
    @test !wrong.warm_start_used && wrong.warm_start_status == :initial_mismatch
    one = product_state(balanced_tree(1),[[1,0]])
    zeroop = SumOfProducts([2],ProductTerm[])
    q,qd = tamen_step(one,zeroop,0.1;warm_start=hint)
    @test !qd.warm_start_used && qd.warm_start_status == :layout_mismatch
    @test dense(q) ≈ dense(one)
    # Diagnostic solutions and hints own independent storage.
    fill!(d.solution.state.cores[d.solution.state.tree.root],0)
    @test dense(hint.solution.state) == stored
    _,failed = tamen_step(x,op,0.3;maxrank=1,maxsweeps=1,strict=false,store_warm_start=true)
    @test !failed.converged && failed.warm_start === nothing
    @test_throws ArgumentError tamen_step(x,op,0.1;warm_start=true)
    tasks = [Threads.@spawn tamen_step(y,op,0.2;warm_start=hint,time_points=5,
        rtol=1e-10,maxrank=8,workspace=TAMEnWorkspace(),threaded=true) for _ in 1:2]
    a,b = fetch.(tasks)
    @test ttn_distance(a[1],b[1]) < 1e-10
    @test dense(hint.solution.state) == stored
    @test_throws TAMEnConvergenceError tamen_step(y,op,0.2;warm_start=hint,
        time_points=5,krylovdim=1,krylov_maxiter=1)
    @test dense(hint.solution.state) == stored
end

@testset "Warm predictor replays increments without extrapolation" begin
    x = product_state(balanced_tree(1),[[1,0]])
    op = SumOfProducts([2],[ProductTerm(1,Dict(1=>ComplexF64[0 0;1 0]))])
    y,d = tamen_step(x,op,0.2;time_points=5,rtol=1e-10,store_warm_start=true)
    for h in (0.1,0.4), nt in (3,7), scheme in (:cheb,:cn)
        grid = TTNSolver._tamen_grid(nt,h,scheme)
        guess,_,status = TTNSolver._tamen_initial_guess(y,grid,d.warm_start)
        nodes = scheme==:cheb ? grid.nodes[2:end] : grid.nodes
        expected = reduce(hcat,([1,0.2+t] for t in nodes))
        @test status == :used
        @test dense(guess) ≈ expected atol=1e-10
    end
end
