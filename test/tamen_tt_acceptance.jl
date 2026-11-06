function tt_acceptance_model(;seed=11)
    rng = MersenneTwister(seed)
    dims = [3,3,3,3]
    local_ops = [-0.4I + 0.3*randn(rng,ComplexF64,3,3) for _ in dims]
    couplings = [0.25*randn(rng,ComplexF64,3,3) for _ in dims]
    terms = [ProductTerm(1,Dict(k=>local_ops[k])) for k in eachindex(dims)]
    for k in 1:length(dims)-1
        push!(terms,ProductTerm(-0.6im,Dict(k=>couplings[k],k+1=>couplings[k+1])))
    end
    push!(terms,ProductTerm(0.2,Dict(1=>couplings[1],4=>couplings[4])))
    op = SumOfProducts(dims,terms)
    tree = BinaryTree([(0,0),(0,0),(1,2),(0,0),(0,0),(4,5),(3,6)])
    x = product_state(tree,[normalize(randn(rng,ComplexF64,3)) for _ in dims])
    return op,x
end

tt_exact(op,x,h) = exp(h*dense(op))*vec(dense(x))
tt_error(y,exact) = norm(vec(dense(y))-exact)/norm(exact)

@testset "TT-style tAMEn acceptance options" begin
    op,x = tt_acceptance_model()
    h = 0.5
    exact = tt_exact(op,x,h)
    common = (;time_points=8,rtol=1e-9,maxrank=9,maxsweeps=20,kickrank=2)

    @testset "truncate! relative budget" begin
        rng = MersenneTwister(5)
        tree = balanced_tree(4)
        leaves = Dict(n=>randn(rng,ComplexF64,3,3) for n in tree.leaf_order)
        # Root (1,6,6) over two internal (6,3,3) cores: bond 6 exceeds the
        # attainable rank 9 only after compression thresholds are applied.
        cores = Dict(n=>(n==tree.root ? randn(rng,ComplexF64,1,6,6) : randn(rng,ComplexF64,6,3,3))
                     for n in eachindex(tree.children) if !(n in tree.leaf_order))
        # A deliberately overcomplete, unnormalized state.
        state = TTNTensor(tree,leaves,cores)
        scale = norm(state)
        for r in (1e-2,1e-4)
            a,b = copy(state),copy(state)
            ia = truncate!(a;atol=0.0,rtol=r)
            ib = truncate!(b;atol=r*scale)
            @test tree_ranks(a) == tree_ranks(b)
            @test dense(a) ≈ dense(b) rtol=1e-10
            @test ia.discarded_bound ≈ ib.discarded_bound rtol=1e-8
            @test ia.tolerance_met
            @test ttn_distance(a,state) <= ia.discarded_bound*(1+1e-8) + 1e-12*scale
        end
        # The absolute budget wins when it is larger.
        a,b = copy(state),copy(state)
        truncate!(a;atol=1e-1*scale,rtol=1e-6)
        truncate!(b;atol=1e-1*scale)
        @test tree_ranks(a) == tree_ranks(b)
        original = dense(state)
        for bad in (-1e-3,NaN,Inf)
            @test_throws ArgumentError truncate!(state;rtol=bad)
        end
        @test dense(state) == original
    end

    @testset "defaults remain certified" begin
        y,info = tamen_step(x,op,h;common...)
        @test info.converged && info.certified
        @test info.global_checks >= 1
        @test isfinite(info.relative_residual_bound)
        @test info.relative_residual_bound <= 1e-9
        @test tt_error(y,exact) < 1e-7
    end

    @testset "certify=false accepts on the auxiliary estimate" begin
        y,info = tamen_step(x,op,h;common...,certify=false)
        @test info.converged
        @test !info.certified
        @test info.global_checks == 0
        @test info.timings.residual_global_seconds == 0
        @test isinf(info.residual_bound) && isinf(info.relative_residual_bound)
        @test all(e->!e.global_residual_checked,info.history)
        @test info.relative_residual_estimate <= 1e-9
        @test tt_error(y,exact) < 1e-6
        @test_throws ArgumentError tamen_step(x,op,h;common...,certify=false,residual_method=:global)
        # Estimate-based failure keeps its own reason and message.
        err = try
            tamen_step(x,op,h;time_points=8,rtol=1e-10,maxrank=1,maxsweeps=1,certify=false)
            nothing
        catch e
            e
        end
        @test err isa TAMEnConvergenceError
        @test err.reason == :residual_estimate
        @test occursin("estimate",sprint(showerror,err))
        _,failed = tamen_step(x,op,h;time_points=8,rtol=1e-10,maxrank=1,maxsweeps=1,
                              certify=false,strict=false)
        @test !failed.converged && !failed.certified
    end

    @testset "truncation_rtol bounds ranks without losing accuracy" begin
        # Truncated directions are not repaired by the final root solve, so the
        # truncation budget must stay below the residual target.
        loose = (;common...,rtol=1e-6)
        _,plain = tamen_step(x,op,h;loose...,certify=false)
        y,cut = tamen_step(x,op,h;loose...,certify=false,truncation_rtol=1e-7)
        @test cut.converged
        plain_ranks = tree_ranks(plain.solution.state)
        cut_ranks = tree_ranks(cut.solution.state)
        @test all(cut_ranks .<= plain_ranks)
        @test sum(cut_ranks) < sum(plain_ranks)
        @test tt_error(y,exact) < 1e-4
        # Certified acceptance still holds after truncation.
        yc,certified = tamen_step(x,op,h;loose...,truncation_rtol=1e-7)
        @test certified.certified && certified.relative_residual_bound <= 1e-6
        @test tt_error(yc,exact) < 1e-4
        for bad in (-1e-3,1.0,NaN)
            @test_throws ArgumentError tamen_step(x,op,h;common...,truncation_rtol=bad)
        end
    end

    @testset "local_tol_factor relaxes local GMRES" begin
        _,tight = tamen_step(x,op,h;common...,certify=false,local_tol_factor=nothing)
        y,loose = tamen_step(x,op,h;common...,certify=false,local_tol_factor=0.01)
        # Defaults: 1% local tolerance and exact channel environments.
        yd,default = tamen_step(x,op,h;common...,certify=false)
        _,explicit = tamen_step(x,op,h;common...,certify=false,local_tol_factor=0.01,local_operator=:channels)
        @test default.matvecs == explicit.matvecs
        @test dense(yd) == dense(tamen_step(x,op,h;common...,certify=false,
            local_tol_factor=0.01,local_operator=:channels)[1])
        @test loose.converged
        @test loose.matvecs < tight.matvecs
        @test tt_error(y,exact) < 1e-6
        # The relaxed local tolerance does not weaken a certificate.
        yc,certified = tamen_step(x,op,h;common...,local_tol_factor=0.01)
        @test certified.certified && certified.relative_residual_bound <= 1e-9
        @test tt_error(yc,exact) < 1e-7
        for bad in (0,0.0,2,NaN,Inf)
            @test_throws ArgumentError tamen_step(x,op,h;common...,local_tol_factor=bad)
        end
    end

    @testset "invariants stay protected with all options" begin
        G = ComplexF64[-1 2; 1 -2]
        S = ComplexF64[1 0.2im; 0.3 1]
        L = S*G/S
        cvec = S' \ ones(ComplexF64,2)
        tree = balanced_tree(2)
        x2 = product_state(tree,[S*ComplexF64[2,0],S*ComplexF64[0.4,0.1]])
        c = product_state(tree,[cvec,cvec])
        op2 = SumOfProducts([2,2],[ProductTerm(1,Dict(1=>L)),ProductTerm(0.7,Dict(2=>L))])
        target = dot(c,x2)
        for als_sweep in (:serial,:etter)
            y,info = tamen_step(x2,op2,0.3;time_points=7,rtol=1e-9,maxrank=4,maxsweeps=10,
                conserved=c,certify=false,truncation_rtol=1e-10,local_tol_factor=0.01,als_sweep)
            @test info.converged && !info.certified
            @test maximum(abs.(info.invariant_values .- target)) < 1e-10
            @test norm(vec(dense(y))-exp(0.3dense(op2))*vec(dense(x2))) < 1e-7
        end
    end

    @testset "warm starts and adaptive intervals" begin
        options = (;certify=false,truncation_rtol=1e-10,local_tol_factor=0.01)
        y1,i1 = tamen_step(x,op,0.25;common...,options...,store_warm_start=true)
        @test !isnothing(i1.warm_start)
        y2,i2 = tamen_step(y1,op,0.25;common...,options...,warm_start=i1.warm_start)
        @test i2.warm_start_used && i2.converged && !i2.certified
        @test tt_error(y2,exact) < 1e-6
        result = tamen_evolve(x,op,(0.0,h);dt=0.25,saveat=[0.0,h],rtol=1e-6,atol=1e-9,
            solve_rtol=1e-9,maxrank=9,kickrank=2,options...)
        @test result.accepted >= 2
        @test all(d->!d.accepted || (d.solve.converged && !d.solve.certified),result.diagnostics)
        @test tt_error(result.final_state,exact) < 1e-5
    end
end
