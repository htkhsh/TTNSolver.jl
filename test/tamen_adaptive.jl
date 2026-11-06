@testset "Adaptive tAMEn intervals and rollback" begin
    L = ComplexF64[-1.5 0.3;0 -0.2]
    op = SumOfProducts([2],[ProductTerm(1,Dict(1=>L))])
    x = product_state(balanced_tree(1),[ComplexF64[1,0.2im]])
    for scheme in (:cheb,:cn)
        nt = scheme==:cheb ? 3 : 5
        work = TAMEnWorkspace()
        result = tamen_evolve(x,op,(2.,3.);dt=1.,saveat=[2.,2.17,2.49,3.],
            time_scheme=scheme,time_points=nt,rtol=2e-5,atol=1e-9,
            time_error_damp=10.,workspace=work,observe=s->vec(dense(s)))
        @test result.accepted > 1
        @test result.rejected > 0
        @test result.times == [2.,2.17,2.49,3.]
        for (t,value) in zip(result.times,result.values)
            @test norm(value-exp((t-2)*L)*vec(dense(x))) < 2e-4
        end
        sparse = tamen_evolve(x,op,(2.,3.);dt=1.,time_scheme=scheme,time_points=nt,
            rtol=2e-5,atol=1e-9,time_error_damp=10.)
        @test sparse.accepted_times == result.accepted_times
        @test ttn_distance(sparse.final_state,result.final_state) < 1e-11
        @test all(d->!d.accepted || d.solve.converged,result.diagnostics)
        @test all(d->!d.accepted || d.error_ratio<=1,result.diagnostics)
        @test all(d->!hasproperty(d.solve,:solution),result.diagnostics)
    end
    for limits in ((;maxsteps=1),(;maxreject=1),(;dtmin=0.9))
        failure = try
            tamen_evolve(x,op,(0.,1.);dt=1.,time_points=2,rtol=1e-12,atol=0.,limits...)
            nothing
        catch e
            e
        end
        @test failure isa TAMEnIntegrationError
        @test failure.time == 0
        @test dense(failure.state) == dense(x)
        @test !isempty(failure.diagnostics)
    end
    result = tamen_evolve(x,op,(0.,0.2);dt=0.2,saveat=[0.,0.03,0.2],
        store_solutions=true,observe=s->(fill!(s.leaves[1],0);nothing))
    @test vec(dense(result.final_state)) ≈ exp(0.2L)*vec(dense(x)) atol=1e-8
    @test hasproperty(last(result.diagnostics).solve,:solution)
    z = product_state(x.tree,[zeros(2)])
    zr = tamen_evolve(z,op,(0.,1.);dt=0.1,atol=0.,rtol=1e-6)
    @test norm(zr.final_state)==0 && zr.rejected==0
    empty_run = tamen_evolve(x,op,(1.,1.);dt=0.1)
    @test empty_run.accepted==0 && isempty(empty_run.diagnostics)
    @test_throws ArgumentError tamen_evolve(x,op,(0.,1.);dt=0.)
    @test_throws ArgumentError tamen_evolve(x,op,(0.,1.);dt=1.,time_error_damp=-1)
    @test_throws ArgumentError tamen_evolve(x,op,(0.,1.);dt=1.,strict=false)
end
