@testset "tAMEn multiple invariants and work ownership" begin
    G = ComplexF64[-1 2 0;1 -2 0;0 0 0]
    S = ComplexF64[1 0.1im 0.2;0.3 1 0;0 0.2im 1]
    tree = balanced_tree(2)
    op = SumOfProducts([3,2],[ProductTerm(1,Dict(1=>S*G/S))])
    x = product_state(tree,[S*ComplexF64[1,0,0.2],[1,0.3im]])
    cs = [product_state(tree,[S'\v,ComplexF64[1,0]]) for v in ([1,1,0],[0,0,1])]
    work = TAMEnWorkspace()
    before = dense(x)
    for scheme in (:cheb,:cn)
        y,d = tamen_step(x,op,0.2;workspace=work,conserved_observables=cs,
            time_scheme=scheme,time_points=7,maxrank=8,rtol=1e-9,threaded=false)
        @test size(d.invariant_history) == (2,length(d.solution.nodes))
        @test d.invariant_history ≈ repeat([dot(c,x) for c in cs],1,length(d.solution.nodes)) atol=1e-10
        yp,p = tamen_step(x,op,0.2;workspace=work,conserved_observables=cs,
            time_scheme=scheme,time_points=7,maxrank=8,rtol=1e-9,threaded=true)
        @test ttn_distance(y,yp) < 1e-11
        @test p.workspace_stats.buffer_reuses > 0
        @test p.workspace_stats.grid_hits == 1
        @test p.threaded == (Threads.nthreads(:default)>1)
    end
    @test dense(x) == before
    @test_throws ArgumentError tamen_step(x,op,0.2;conserved=cs[1],conserved_observables=cs)
    @test_throws ArgumentError tamen_step(x,op,0.2;conserved_observables=cs,maxrank=1)
    @test_throws TAMEnConvergenceError tamen_step(x,op,0.2;workspace=work,krylovdim=1,krylov_maxiter=1)
    @test tamen_step(x,op,0.2;workspace=work)[2].converged # lease released on error
    busy = TTNSolver._with_tamen_workspace(work) do active
        @test_throws ArgumentError tamen_step(x,op,0.2;workspace=active)
        true
    end
    @test busy
    tasks = [Threads.@spawn tamen_step(x,op,0.2;workspace=TAMEnWorkspace(),threaded=true) for _ in 1:2]
    a,b = fetch.(tasks)
    @test ttn_distance(a[1],b[1]) < 1e-11
    zeroop = SumOfProducts([3,2],ProductTerm[])
    _,z = tamen_step(x,zeroop,0.2;estimate_time_error=true)
    @test z.time_error.absolute < 1e-11
end
