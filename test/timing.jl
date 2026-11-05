@testset "Adaptive timing preserves trajectories" begin
    initial = product_state(balanced_tree(3),
        [ComplexF64[1,0],ComplexF64[1,1]/sqrt(2),ComplexF64[1,im]/sqrt(2)])
    X = ComplexF64[0 1;1 0]
    Z = ComplexF64[1 0;0 -1]
    op = SumOfProducts([2,2,2], [ProductTerm(-0.3im,Dict(1=>X,2=>Z)),
                                ProductTerm(-0.2im,Dict(2=>X,3=>X))])
    options = (;dt=0.05,saveat=0:0.1:0.2,atol=1e-8,rtol=1e-4,observe=norm)
    plain = adaptive_evolve(initial,op,(0.,0.2);options...)
    timed = adaptive_evolve(initial,op,(0.,0.2);options...,record_timing=true)
    @test plain.timings === nothing
    @test all(d -> d.timings === nothing,plain.diagnostics)
    @test timed.times == plain.times
    @test timed.stats == plain.stats
    @test timed.values ≈ plain.values atol=1e-14
    @test dense(timed.final_state) ≈ dense(plain.final_state) atol=1e-14
    @test [(d.dt,d.accepted,d.reason) for d in timed.diagnostics] ==
          [(d.dt,d.accepted,d.reason) for d in plain.diagnostics]
    tm = timed.timings
    @test tm.coarse_bug > 0 && tm.accepted_update > 0 && tm.difference > 0
    @test tm.observations > 0 && tm.candidate_norm > 0
    @test all(>=(0),values(tm))
    @test tm.total ≈ tm.coarse_bug + tm.accepted_update + tm.rejected_update +
        tm.difference + tm.candidate_norm + tm.observations + tm.other
    @test tm.coarse_bug ≈ sum(d.timings.coarse_bug for d in timed.diagnostics)
    @test tm.accepted_update ≈ sum(d.timings.fine_bug for d in timed.diagnostics if d.accepted)
    @test tm.rejected_update ≈ sum((d.timings.fine_bug for d in timed.diagnostics if !d.accepted);init=0.0)

    zero = adaptive_evolve(initial,op,(0.,0.);dt=0.1,record_timing=true,observe=norm)
    @test zero.timings.coarse_bug == zero.timings.accepted_update == zero.timings.difference == 0
    @test zero.timings.observations > 0
    @test isempty(zero.diagnostics)
end

@testset "Timing includes aborted and rejected work" begin
    A = ComplexF64[-1 1 0 0;0 -2 1 0;0 0 -3 1;1 0 0 -4]
    initial = product_state(balanced_tree(1),[[1,0,0,0]])
    op = SumOfProducts([4],[ProductTerm(1,Dict(1=>A))])
    result = adaptive_evolve(initial,op,(0.,0.05);dt=0.05,record_timing=true,
        krylovdim=2,krylov_maxiter=1,krylov_tol=0.01,atol=1e-3,rtol=0)
    @test result.stats.rejected > 0
    failed = filter(d -> d.reason == :local_solve,result.diagnostics)
    @test !isempty(failed)
    @test all(d -> d.timings.coarse_bug + d.timings.fine_bug > 0,failed)
    @test all(d -> d.timings.difference == 0,failed)

    tree = balanced_tree(2)
    entangled = TTNTensor(tree,Dict(n=>Matrix{ComplexF64}(I,2,2) for n in tree.leaf_order),
        Dict(tree.root=>reshape(Matrix{ComplexF64}(I,2,2),1,2,2)))
    err = try
        adaptive_evolve(entangled,SumOfProducts([2,2],ProductTerm[]),(0.,1.);
            dt=0.1,maxrank=1,maxreject=2,record_timing=true)
        nothing
    catch e
        e
    end
    @test err isa AdaptiveIntegrationError
    if err isa AdaptiveIntegrationError
        @test all(d -> d.stage == :coarse && !d.accepted,err.diagnostics)
        @test all(d -> d.timings.coarse_bug > 0,err.diagnostics)
        @test all(d -> d.timings.fine_bug == d.timings.difference == d.timings.candidate_norm == 0,
                  err.diagnostics)
    end
end
