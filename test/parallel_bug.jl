# A two-leaf example with hand-derived old/new blocks distinguishes parallel
# BUG's explicit cross blocks from an augmented Galerkin exponential.
@testset "Parallel BUG" begin
    @test isdefined(TTNSolver, :parallel_bug_step)
    if isdefined(TTNSolver, :parallel_bug_step)
        tree = balanced_tree(2)
        x = product_state(tree, [[1, 0], [1, 0]])
        flip = ComplexF64[0 1; 1 0]
        op = SumOfProducts([2,2], [ProductTerm(-0.3im, Dict(1=>flip)),
            ProductTerm(-0.2im, Dict(2=>flip)), ProductTerm(-im, Dict(1=>flip,2=>flip))])
        for threaded in (false,true)
            y, info = parallel_bug_step(x,op,0.1;atol=0,threaded)
            @test dense(y) ≈ ComplexF64[1 -0.02im; -0.03im 0] atol=2e-14
            @test info.omitted_block_norms[tree.root] ≈ 0.1 atol=1e-14
            @test info.local_solves == 3
            @test dense(x) == ComplexF64[1 0;0 0]
        end

        rng = MersenneTwister(510)
        A = ComplexF64[-1 2im;0.3 -0.4im]
        B = ComplexF64[0 0.7;-0.2im -0.3]
        for tree in (balanced_tree(3),
            BinaryTree([(2,5),(0,0),(0,0),(0,0),(4,3)];root=1,leaf_order=[3,2,4]))
            x = full_state(tree,[2,2,2],rng)
            op = SumOfProducts([2,2,2],[ProductTerm(1,Dict(1=>A,3=>B)),
                ProductTerm(0.1im,Dict(2=>A)), ProductTerm(-0.2,Dict())])
            y,info = parallel_bug_step(x,op,0.05;atol=0,threaded=false)
            z,threadinfo = parallel_bug_step(x,op,0.05;atol=0,threaded=true)
            @test vec(dense(y)) ≈ exp(0.05*dense(op))*vec(dense(x)) atol=2e-11
            @test dense(z) ≈ dense(y) atol=2e-13
            @test info.local_solves == threadinfo.local_solves
            @test info.skipped_leaf_solves == 3
            test_root_gauge(y)
        end

        @testset "First-order convergence on a deep tree" begin
            tree = balanced_tree(5)
            x = product_state(tree,[ComplexF64[1,0.2im] for _ in 1:5])
            op = SumOfProducts(fill(2,5),[ProductTerm(-im,Dict(j=>A)) for j in 1:5])
            reference = exp(0.2*dense(op))*vec(dense(x))
            errors = Float64[]
            for h in (0.02,0.01)
                y = copy(x)
                for _ in 1:round(Int,0.2/h)
                    y,_ = parallel_bug_step(y,op,h;atol=1e-13,threaded=false)
                end
                push!(errors,norm(vec(dense(y))-reference))
            end
            @test errors[1] > 1e-9
            @test errors[2] < 0.65errors[1]
        end

        @testset "Conservation before correction and small amplitudes" begin
            tree = balanced_tree(3)
            # c is initially absent from each leaf space; c'L=0.
            x = product_state(tree,[[1,0] for _ in 1:3])
            c = product_state(tree,[[1,1] for _ in 1:3])
            jump = ComplexF64[-1 0.3;1 -0.3]
            op = SumOfProducts(fill(2,3),[ProductTerm(1,Dict(j=>jump)) for j in 1:3])
            for scale in (1.0,1e-18)
                tiny = copy(x); tiny.cores[tree.root] .*= scale
                y,info = parallel_bug_step(tiny,op,0.1;atol=0,conserved=c,correct_invariant=false)
                @test dot(c,y)/scale ≈ 1 atol=2e-13
                @test info.invariant_uncorrected/scale ≈ 1 atol=2e-13
                @test iszero(info.correction_norm)
            end
        end

        @testset "Zero, singleton, and local failure" begin
            for n in (1,3)
                tree = balanced_tree(n)
                x = product_state(tree,[zeros(2) for _ in 1:n])
                op = SumOfProducts(fill(2,n),[ProductTerm(1,Dict(1=>A))])
                y,_ = parallel_bug_step(x,op,0.2;atol=0)
                @test iszero(norm(y))
            end
            x = product_state(balanced_tree(1),[ComplexF64[1,im]])
            op = SumOfProducts([2],[ProductTerm(1,Dict(1=>A))])
            y,_ = parallel_bug_step(x,op,0.1)
            @test dense(y) ≈ exp(0.1A)*dense(x) atol=1e-12
            @test_throws ArgumentError parallel_bug_step(x,op,0)
            @test_throws ArgumentError parallel_bug_step(x,op,0.1;krylovdim=0)
            x = product_state(balanced_tree(3),[ComplexF64[1,im] for _ in 1:3])
            op = SumOfProducts(fill(2,3),[ProductTerm(1,Dict(j=>A)) for j in 1:3])
            before = dense(x)
            for threaded in (false,true)
                @test_throws LocalSolveFailure parallel_bug_step(x,op,5.0;
                    threaded,krylovdim=1,krylov_maxiter=1,krylov_tol=1e-14)
                @test dense(x) == before
            end
        end

        @testset "Forty leaves without expansion" begin
            tree = balanced_tree(40)
            x = product_state(tree,[ComplexF64[1,0] for _ in 1:40])
            op = SumOfProducts(fill(2,40),[ProductTerm(1,Dict(1=>A)),ProductTerm(1,Dict(40=>B))])
            y,info = parallel_bug_step(x,op,0.01;atol=1e-12)
            z,other = parallel_bug_step(x,op,0.01;atol=1e-12,threaded=false)
            @test ttn_distance(y,z) < 1e-12
            @test info.max_local_dimension <= 8
            @test norm(y) > 0.9
        end
    end
end
