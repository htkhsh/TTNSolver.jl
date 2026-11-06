function projection_reference(sol)
    # Orthonormal basis of the spatial span through the collocation samples.
    samples = hcat([vec(dense(time_slice(sol,j))) for j in 2:length(sol.nodes)]...)
    F = svd(samples)
    keep = count(>(1e-12*F.S[1]),F.S)
    Q = F.U[:,1:keep]
    x0 = vec(dense(sol.initial))
    return norm(x0-Q*(Q'*x0)),keep
end

function projection_coefficients(op,sol)
    st = sol.state
    nt = size(st.leaves[sol.time_leaf],1)
    lifted = TTNSolver._time_lift(sol.initial,ones(nt))
    spacetime = SumOfProducts([op.dimensions;nt],op.terms)
    _,cross = TTNSolver._tamen_upward(spacetime,st,lifted)
    return vec(cross[sol.initial.tree.root])
end

old_projection_error(sol,u0) = ttn_distance(sol.initial,TTNSolver._spatial_coefficients(sol,u0))

@testset "Cancellation-free initial projection error" begin
    op,x = tt_acceptance_model(seed=3)
    cases = Any[]
    _,info = tamen_step(x,op,0.4;time_points=6,rtol=1e-9,maxrank=9,maxsweeps=20,
        kickrank=2,store_warm_start=true,estimate_time_error=true)
    push!(cases,("cold",info))
    y,_ = tamen_step(x,op,0.2;time_points=6,rtol=1e-9,maxrank=9,maxsweeps=20,kickrank=2)
    _,warm = tamen_step(y,op,0.2;time_points=6,rtol=1e-9,maxrank=9,maxsweeps=20,kickrank=2,
        warm_start=info.warm_start,estimate_time_error=true)
    push!(cases,("warm",warm))
    # A coarse truncation leaves a visible component of x0 outside the basis.
    _,cut = tamen_step(x,op,0.4;time_points=6,rtol=1e-4,maxrank=9,maxsweeps=20,kickrank=2,
        certify=false,truncation_rtol=2e-5,estimate_time_error=true)
    push!(cases,("truncated",cut))
    for (label,d) in cases
        sol = d.solution
        u0 = projection_coefficients(op,sol)
        new = TTNSolver._tamen_projection_error(sol,u0)
        old = old_projection_error(sol,u0)
        reference,keep = projection_reference(sol)
        scale = norm(sol.initial)
        @test d.time_error.initial_projection_error == new
        @test abs(new-old) <= 1e-12*scale + 1e-8*old
        # The dense sample span equals the solution span only at full root rank.
        if keep == size(sol.state.cores[sol.state.tree.root],2)
            @test abs(new-reference) <= 1e-12*scale + 1e-8*reference
        end
    end
    @test cases[3][2].time_error.initial_projection_error > 1e-10

    @testset "single-leaf spatial tree" begin
        L = ComplexF64[-0.4 0.8im 0; 0.2 -1.3 0.1; 0 0.3im -0.2]
        op1 = SumOfProducts([3],[ProductTerm(1,Dict(1=>L))])
        x1 = product_state(balanced_tree(1),[ComplexF64[2,0.3im,0.1]])
        _,d = tamen_step(x1,op1,1.0;time_points=4,rtol=1e-10,maxrank=3,estimate_time_error=true)
        u0 = projection_coefficients(op1,d.solution)
        @test d.time_error.initial_projection_error ≈ old_projection_error(d.solution,u0) atol=1e-13
    end

    @testset "nonorthonormal bases use the exact fallback" begin
        sol = cases[1][2].solution
        u0 = projection_coefficients(op,sol)
        state = copy(sol.state)
        leaf = first(sol.initial.tree.leaf_order)
        state.leaves[leaf] = 2 .* state.leaves[leaf]
        skewed = SpaceTimeTTN(state,sol.initial,sol.nodes,sol.time_leaf,sol.time_scheme)
        @test TTNSolver._tamen_projection_error(skewed,u0) == old_projection_error(skewed,u0)
    end
end

@testset "Time error reuses the final root environments" begin
    op,x = tt_acceptance_model(seed=6)
    grid = TTNSolver._tamen_grid(6,0.4,:cheb)
    A = TTNSolver._spacetime_operator(op,grid)
    # Chebyshev spatial terms carry no explicit identity time factor.
    @test all(t->!haskey(t.factors,length(op.dimensions)+1),A.terms[2:end])
    @test dense(A) ≈ kron(grid.D,Matrix{ComplexF64}(I,size(dense(op))...)) -
        kron(Matrix{ComplexF64}(I,6,6),dense(op))
    cn = TTNSolver._spacetime_operator(op,TTNSolver._tamen_grid(5,0.4,:cn))
    @test all(t->haskey(t.factors,length(op.dimensions)+1),cn.terms[2:end])
    for lo in (:sop,:channels), scheme in (:cheb,:cn)
        points = scheme == :cheb ? 6 : 5
        _,d = tamen_step(x,op,0.4;time_points=points,time_scheme=scheme,rtol=1e-9,maxrank=9,
            kickrank=2,estimate_time_error=true,local_operator=lo)
        recomputed = TTNSolver._tamen_time_error(d.solution,op;local_operator=lo)
        @test d.time_error.absolute ≈ recomputed.absolute rtol=1e-8 atol=1e-14
        @test d.time_error.initial_projection_error ≈ recomputed.initial_projection_error rtol=1e-8 atol=1e-15
    end
end
