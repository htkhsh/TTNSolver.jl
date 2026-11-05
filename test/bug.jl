function full_state(tree, dims, rng)
    leaves = Dict(node => Matrix{ComplexF64}(I, n, n) for (node, n) in zip(tree.leaf_order, dims))
    cores = Dict{Int,Array{ComplexF64,3}}()
    ranks = Dict(node => n for (node, n) in zip(tree.leaf_order, dims))
    for node in tree.postorder
        haskey(leaves, node) && continue
        l, r = tree.children[node]
        a, b = ranks[l], ranks[r]
        ranks[node] = node == tree.root ? 1 : a*b
        cores[node] = node == tree.root ? randn(rng, ComplexF64, 1, a, b) :
            reshape(Matrix{ComplexF64}(I, a*b, a*b), a*b, a, b)
    end
    return TTNTensor(tree, leaves, cores)
end

@testset "Recursive augmented BUG" begin
    rng = MersenneTwister(317)
    A = ComplexF64[-1 2im; 0.3 -0.4im]
    B = ComplexF64[0 0.7; -0.2im -0.3]
    for tree in (balanced_tree(3),
                 BinaryTree([(2, 5), (0, 0), (0, 0), (0, 0), (4, 3)]; root=1, leaf_order=[3, 2, 4]))
        x = full_state(tree, [2, 2, 2], rng)
        op = SumOfProducts([2, 2, 2], [ProductTerm(1, Dict(1=>A, 3=>B)),
                                     ProductTerm(0.1im, Dict(2=>A)), ProductTerm(-0.2, Dict())])
        before = dense(x)
        y, info = augmented_bug_step(x, op, 0.05; atol=0)
        @test vec(dense(y)) ≈ exp(0.05*dense(op)) * vec(before) atol=2e-11
        @test dense(x) == before
        @test info.skipped_leaf_solves == 3
        @test info.truncation.tolerance_met
        test_root_gauge(y)
    end
    # First derivative creates a second Schmidt direction from a product.
    tree = balanced_tree(2)
    x = product_state(tree, [[1, 0], [1, 0]])
    flip = ComplexF64[0 1; 1 0]
    op = SumOfProducts([2, 2], [ProductTerm(-im, Dict(1=>flip, 2=>flip))])
    # With zero old contractions this example has no BUG basis seed. Add a
    # local term so the K stages expose the coupling direction on both leaves.
    op = SumOfProducts([2, 2], [op.terms; ProductTerm(-0.3im, Dict(1=>flip));
                             ProductTerm(-0.2im, Dict(2=>flip))])
    y, info = augmented_bug_step(x, op, 0.1; atol=1e-12)
    @test maximum(tree_ranks(y)) == 2
    @test vec(dense(y)) ≈ exp(0.1*dense(op))*vec(dense(x)) atol=1e-11
    @test info.local_solves == 3

    # Arbitrarily small amplitude must survive the basis construction.
    tiny = copy(x)
    tiny.cores[tree.root] .*= 1e-18
    z, _ = augmented_bug_step(tiny, op, 0.1; atol=0)
    @test vec(dense(z))/1e-18 ≈ vec(dense(y)) atol=1e-11
    near_singular = TTNTensor(tree,
        Dict(n=>Matrix{ComplexF64}(I, 2, 2) for n in tree.leaf_order),
        Dict(tree.root=>reshape(diagm(ComplexF64[1, 1e-16]), 1, 2, 2)))
    evolved, _ = augmented_bug_step(near_singular, op, 0.1; atol=0)
    @test vec(dense(evolved)) ≈ exp(0.1*dense(op))*vec(dense(near_singular)) atol=1e-12
    for count in (1, 3)
        tree = balanced_tree(count)
        x = product_state(tree, [ComplexF64[1, im] for _ in 1:count])
        zero = SumOfProducts(fill(2, count), ProductTerm[])
        y, _ = augmented_bug_step(x, zero, 0.2; atol=0)
        @test dense(y) ≈ dense(x) atol=1e-12
        z = product_state(tree, [zeros(2) for _ in 1:count])
        yz, _ = augmented_bug_step(z, zero, 0.2)
        @test iszero(dense(yz))
    end
    single = product_state(balanced_tree(1), [ComplexF64[1, im]])
    op = SumOfProducts([2], [ProductTerm(1, Dict(1=>A))])
    y, _ = augmented_bug_step(single, op, 0.1)
    @test dense(y) ≈ exp(0.1*A)*dense(single) atol=1e-12
    @test_throws ArgumentError augmented_bug_step(single, op, 0)
    @test_throws ArgumentError augmented_bug_step(single, op, NaN)
    @test_throws ArgumentError augmented_bug_step(single, op, 0.1; krylov_tol=0)
    @test_throws ArgumentError augmented_bug_step(single, op, 0.1; krylovdim=0)
    @test_logs begin
        @test_throws LocalSolveFailure augmented_bug_step(single, op, 5.0;
            krylovdim=1, krylov_maxiter=1, krylov_tol=1e-14)
    end

    @testset "Forty leaves without global expansion" begin
        tree = balanced_tree(40)
        vectors = [ComplexF64[1, 0] for _ in 1:40]
        x = product_state(tree, vectors)
        L = SumOfProducts(fill(2, 40), [ProductTerm(1, Dict(1=>A)), ProductTerm(1, Dict(40=>B))])
        y, info = augmented_bug_step(x, L, 0.1; atol=1e-12)
        vectors[1] = exp(0.1A)*vectors[1]
        vectors[40] = exp(0.1B)*vectors[40]
        reference = product_state(tree, vectors)
        @test norm(y) ≈ norm(reference) atol=1e-12
        @test dot(reference, y) ≈ dot(reference, reference) atol=1e-12
        @test info.max_local_dimension <= 8
    end

    @testset "Invariant scaling cannot lose a represented direction" begin
        for count in (1, 3), scale in (1e-170, 1e170)
            tree = balanced_tree(count)
            x = product_state(tree, [ComplexF64[1, 0] for _ in 1:count])
            factors = [ComplexF64[1, 0] for _ in 1:count]
            factors[1] .*= scale
            c = product_state(tree, factors)
            op = SumOfProducts(fill(2, count), ProductTerm[])
            y, info = augmented_bug_step(x, op, 0.1; conserved=c, atol=0)
            @test dense(y) ≈ dense(x) atol=1e-14
            @test info.invariant_after / scale ≈ 1 atol=1e-14
            @test isfinite(info.correction_norm)
        end
    end
end
