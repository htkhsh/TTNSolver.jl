@testset "Accepted-only adaptive warm starts" begin
    L = ComplexF64[-1.5 0.3;0 -0.2]
    op = SumOfProducts([2],[ProductTerm(1,Dict(1=>L))])
    x = product_state(balanced_tree(1),[ComplexF64[1,0.2im]])
    work = TAMEnWorkspace()
    for scheme in (:cheb,:cn)
        nt = scheme==:cheb ? 3 : 5
        options = (;dt=1.,time_scheme=scheme,time_points=nt,rtol=2e-5,atol=1e-9,
                   time_error_damp=10.,solve_rtol=1e-10)
        result = tamen_evolve(x,op,(0.,1.);options...,workspace=work,warm_start=true)
        cold = tamen_evolve(x,op,(0.,1.);options...,warm_start=false)
        @test result.accepted>1 && result.rejected>0
        @test norm(vec(dense(result.final_state))-exp(L)*vec(dense(x))) < 2e-4
        @test ttn_distance(result.final_state,cold.final_state) < 2e-4
        had_accepted = false
        for attempt in result.diagnostics
            @test attempt.solve.warm_start_used == had_accepted
            @test !hasproperty(attempt.solve,:warm_start)
            @test !hasproperty(attempt.solve,:solution)
            had_accepted |= attempt.accepted
        end
        @test all(d->!d.solve.warm_start_used,cold.diagnostics)
        @test !first(result.diagnostics).solve.warm_start_used # workspace has no hidden trajectory
    end
    saved = tamen_evolve(x,op,(0.,0.3);dt=0.1,dtmax=0.1,time_error_damp=0,
        store_solutions=true,warm_start=true)
    @test all(d->hasproperty(d.solve,:solution) && !hasproperty(d.solve,:warm_start),saved.diagnostics)
    @test all(d->d.solve.warm_start_used,saved.diagnostics[2:end])
    failure = try
        tamen_evolve(x,op,(0.,1.);dt=1.,time_points=2,rtol=1e-12,atol=0.,maxreject=1)
        nothing
    catch e
        e
    end
    @test failure isa TAMEnIntegrationError
    @test failure.time == 0. && dense(failure.state) == dense(x)
    @test !hasproperty(first(failure.diagnostics).solve,:warm_start)
    @test_throws ArgumentError tamen_evolve(x,op,(0.,0.1);dt=0.1,store_warm_start=true)
end
