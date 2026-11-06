@testset "Space-time TTN AMEn prototype" begin
    @testset "Chebyshev initial condition elimination" begin
        for nt in (2, 5, 10), h in (0.1, 2.0)
            grid = chebyshev_collocation(nt, h)
            @test first(grid.nodes) == 0
            @test last(grid.nodes) == h
            @test all(diff(grid.nodes) .> 0)
            @test grid.D * ones(nt) ≈ grid.d0 atol=1e-11
            t = grid.nodes[2:end]
            @test grid.D * (t .^ 2) ≈ 2t atol=1e-11
        end
        @test_throws ArgumentError chebyshev_collocation(0, 1.0)
        @test_throws ArgumentError chebyshev_collocation(4, 0.0)
    end

    @testset "Single spatial leaf and spectral convergence" begin
        L = ComplexF64[-0.4 0.8im; 0.2 -1.3]
        op = SumOfProducts([2], [ProductTerm(1, Dict(1=>L))])
        x = product_state(balanced_tree(1), [ComplexF64[2, 0.3im]])
        original = dense(x)
        errors = Float64[]
        for nt in (3, 8)
            y, info = tamen_step(x, op, 1.0; time_points=nt, rtol=1e-10,
                                maxrank=2, maxsweeps=8)
            @test info.converged
            @test info.relative_residual_bound <= 1e-10
            @test dense(time_slice(info.solution, 1)) == original
            @test dense(time_slice(info.solution, nt+1)) ≈ dense(y)
            @test info.solution.state.leaves[info.solution.time_leaf] ≈ Matrix(I,nt,nt)
            push!(errors, norm(vec(dense(y))-exp(L)*vec(original)))
        end
        @test errors[2] < errors[1]/1000
        @test errors[2] < 1e-8
        @test dense(x) == original
    end

    @testset "Residual enrichment escapes a product state" begin
        sx = ComplexF64[0 1; 1 0]
        sz = ComplexF64[1 0; 0 -1]
        # Nontrivial leaf order and root ID exercise topology-independent lifting.
        tree = BinaryTree([(2,3),(0,0),(4,5),(0,0),(0,0)]; root=1,leaf_order=[5,2,4])
        x = product_state(tree, [[1,0], [1,0], [1,0]])
        op = SumOfProducts([2,2,2], [ProductTerm(-0.7im, Dict(1=>sx,2=>sx)),
             ProductTerm(-0.3im, Dict(2=>sx,3=>sx)), ProductTerm(-0.2, Dict(3=>sz))])
        y, info = tamen_step(x,op,0.4; time_points=8,rtol=1e-9,maxrank=8,maxsweeps=12)
        exact = exp(0.4dense(op))*vec(dense(x))
        @test norm(vec(dense(y))-exact) < 1e-8
        @test maximum(tree_ranks(y)) > 1
        @test info.converged
        st = info.solution.state
        grid = chebyshev_collocation(8,0.4)
        A = kron(grid.D,Matrix{ComplexF64}(I,8,8)) - kron(Matrix{ComplexF64}(I,8,8),dense(op))
        b = kron(grid.d0,vec(dense(x)))
        exact_residual = norm(b-A*vec(dense(st)))
        @test exact_residual <= info.residual_bound + 1e-11
        @test vec(dense(st)) ≈ A\b rtol=1e-8
        @test_throws TAMEnConvergenceError tamen_step(x,op,0.4;
            time_points=8,rtol=1e-10,maxrank=1,maxsweeps=2)
        _, failed = tamen_step(x,op,0.4; time_points=8,rtol=1e-10,
                              maxrank=1,maxsweeps=2,strict=false)
        @test !failed.converged
        @test length(sprint(showerror,TAMEnConvergenceError(:global_residual,failed))) < 500
        @test maximum(tree_ranks(failed.solution.state)[1:5]) <= 1
    end

    @testset "Complex linear invariant at every time node" begin
        G = ComplexF64[-1 2; 1 -2]
        S = ComplexF64[1 0.2im; 0.3 1]
        L = S*G/S
        cvec = S' \ ones(ComplexF64,2)
        tree = balanced_tree(2)
        x = product_state(tree, [S*ComplexF64[2,0], S*ComplexF64[0.4,0.1]])
        c = product_state(tree, [cvec,cvec])
        op = SumOfProducts([2,2], [ProductTerm(1,Dict(1=>L)),ProductTerm(0.7,Dict(2=>L))])
        target = dot(c,x)
        y, info = tamen_step(x,op,0.3;time_points=7,rtol=1e-9,maxrank=4,
                            maxsweeps=10,conserved=c)
        @test maximum(abs.(info.invariant_values .- target)) < 1e-10
        @test dot(c,y) ≈ target atol=1e-10
        @test norm(vec(dense(y))-exp(0.3dense(op))*vec(dense(x))) < 1e-8
    end

    @testset "Zero generator and validation" begin
        x = product_state(balanced_tree(2), [[2,0],[1,1im]])
        zeroop = SumOfProducts([2,2],ProductTerm[])
        y, info = tamen_step(x,zeroop,0.5; maxrank=1,time_points=6)
        @test dense(y) ≈ dense(x) atol=1e-12
        @test info.converged
        z = product_state(x.tree, [[0,0],[1,0]])
        yz, iz = tamen_step(z,zeroop,0.5)
        @test norm(yz) == 0
        @test iz.converged
        @test_throws ArgumentError tamen_step(x,zeroop,-1)
        @test_throws ArgumentError tamen_step(x,zeroop,1;rtol=-1)
        @test_throws ArgumentError tamen_step(x,zeroop,1;rtol=big"1e-400")
        unlimited, ui = tamen_step(x,zeroop,1;maxrank=typemax(Int))
        @test ui.converged && dense(unlimited) ≈ dense(x)
        @test_throws ArgumentError tamen_step(x,zeroop,1;maxsweeps=0)
        @test_throws DimensionMismatch tamen_step(x,SumOfProducts([3],ProductTerm[]),1)
    end
end

@testset "Pinned time basis and residual compression bounds" begin
    rng = MersenneTwister(814)
    tree = balanced_tree(3)
    x = TTNTensor(tree,Dict(n=>randn(rng,ComplexF64,3,2) for n in tree.leaf_order),
        Dict(n=>randn(rng,ComplexF64,n==tree.root ? 1 : 2,2,2)
             for n in tree.postorder if !TTNSolver._isleaf(tree,n)))
    st = TTNSolver._time_lift(x,ones(6);full_time_basis=true)
    before = dense(st)
    time_leaf = length(tree.children)+1
    cut = truncate!(st;maxrank=1,atol=0,fixed_leaves=[time_leaf])
    @test size(st.leaves[time_leaf],2) == 6
    @test norm(dense(st)-before) <= cut.discarded_bound + 1e-10
    @test maximum(tree_ranks(st)[1:length(tree.children)]) <= 1
    @test_throws ArgumentError truncate!(st;fixed_leaves=[st.tree.root])
    op = SumOfProducts([3,3,3],[ProductTerm(0.4im,
        Dict(1=>randn(rng,ComplexF64,3,3),3=>randn(rng,ComplexF64,3,3))),
        ProductTerm(-0.2,Dict(2=>randn(rng,ComplexF64,3,3)))])
    grid = chebyshev_collocation(4,0.3)
    A = TTNSolver._spacetime_operator(op,grid)
    st = TTNSolver._time_lift(x,ones(4);full_time_basis=true)
    b = TTNSolver._time_lift(x,grid.d0)
    residual,bound,dropped = TTNSolver._tamen_residual(A,st,b,1e-12,1)
    exact = vec(dense(b))-dense(A)*vec(dense(st))
    @test dropped > 1e-12 # a rank cap must not conceal residual assembly error
    @test norm(exact) <= bound + 1e-9
    @test norm(exact-vec(dense(residual))) <= dropped + 1e-9
    # Insufficient GMRES work must throw, even in diagnostic non-strict mode.
    @test_throws TAMEnConvergenceError tamen_step(x,op,0.3;time_points=4,
        krylovdim=1,krylov_maxiter=1,strict=false)
end
