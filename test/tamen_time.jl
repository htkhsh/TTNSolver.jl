@testset "tAMEn time schemes and dense output" begin
    L = ComplexF64[-0.3 0.8im;0.2 -0.7]
    op = SumOfProducts([2],[ProductTerm(1,Dict(1=>L))])
    x = product_state(balanced_tree(1),[ComplexF64[1,0.3im]])
    for scheme in (:cheb,:CN), nt in (5,9)
        y,d = tamen_step(x,op,0.7;time_scheme=scheme,time_points=nt,
                         rtol=1e-11,maxrank=3,estimate_time_error=true)
        sol = d.solution
        @test sol.time_scheme == (scheme == :CN ? :cn : scheme)
        @test length(sol.nodes) == (scheme == :CN ? nt : nt+1)
        if scheme == :CN
            h = 0.7/(nt-1)
            reference = ((I-h*L/2)\(I+h*L/2))^(nt-1)*vec(dense(x))
            @test vec(dense(y)) ≈ reference atol=1e-10
        end
        @test dense(time_snapshot(sol,0.0)) == dense(x)
        @test dense(time_snapshot(sol,0.7)) ≈ dense(y)
        @test d.time_error.absolute >= 0
        @test isfinite(d.time_error.relative)
        @test d.time_error.fine_points > length(sol.nodes)
        @test_throws ArgumentError time_snapshot(sol,-0.1)
        @test_throws ArgumentError time_snapshot(sol,0.71)
        @test_throws ArgumentError time_snapshot(sol,NaN)
    end
    _,lo = tamen_step(x,op,0.7;time_points=3,rtol=1e-11,estimate_time_error=true)
    _,hi = tamen_step(x,op,0.7;time_points=8,rtol=1e-11,estimate_time_error=true)
    @test hi.time_error.absolute < lo.time_error.absolute/1000
    @test norm(vec(dense(time_snapshot(hi.solution,0.37)))-exp(0.37L)*vec(dense(x))) < 1e-9
    @test_throws ArgumentError tamen_step(x,op,1;time_scheme=:invalid)
    @test_throws ArgumentError tamen_step(x,op,1;time_scheme=:cn,time_points=4)
    @test_throws ArgumentError tamen_step(x,op,1;time_scheme=:cn,time_points=1)
    # A valid time extremely close to the initial node must not overflow
    # barycentric reciprocals (also used by adaptive saveat).
    zeroop = SumOfProducts([2],ProductTerm[])
    _,constant = tamen_step(x,zeroop,1.;time_points=5)
    @test dense(time_snapshot(constant.solution,1e-310)) ≈ dense(x) atol=1e-12
    @test dense(time_snapshot(constant.solution,nextfloat(0.))) ≈ dense(x) atol=1e-12
end
