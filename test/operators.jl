@testset "Product-sum operators" begin
    a = ComplexF64[1 im; 2 -0.5]
    b = ComplexF64[1 0 im; 2 3 0; -im 1 2]
    c = ComplexF64[0 2im; 1 1]
    @testset "Complex action and matrix elements" begin
        for order in ([1, 2, 3], [3, 1, 2])
            tree, leaves, cores, _ = three_leaf_fixture(; order)
            x = TTNTensor(tree, leaves, cores)
            yt, yl, yc, _ = three_leaf_fixture(; order, ranks=(1, 3, 1, 2), seed=914)
            y = TTNTensor(yt, yl, yc)
            dims = leafdims(x)
            factors = Dict(1 => a, findfirst(==(3), dims) => b)
            # Physical positions, not node IDs, determine local factor placement.
            op = SumOfProducts(dims, [ProductTerm(2-im, factors),
                ProductTerm(-0.3im, Dict(length(dims) => (dims[end] == 2 ? c : b))),
                ProductTerm(0.7, Dict())])
            identity_factors = [Matrix{ComplexF64}(I, n, n) for n in dims]
            first_factors = [get(factors, k, identity_factors[k]) for k in 1:3]
            last_factors = copy(identity_factors)
            last_factors[end] = dims[end] == 2 ? c : b
            expected = (2-im) * kron(reverse(first_factors)...) -
                0.3im * kron(reverse(last_factors)...) + 0.7I
            @test dense(op) ≈ expected atol=1e-12 rtol=1e-12
            before_leaves, before_cores = deepcopy(x.leaves), deepcopy(x.cores)
            result = apply(op, x)
            @test vec(dense(result)) ≈ expected * vec(dense(x)) atol=1e-12 rtol=1e-12
            @test dense(op * x) ≈ dense(result) atol=1e-12
            @test dot(y, op, x) ≈ dot(vec(dense(y)), expected * vec(dense(x))) atol=1e-12
            @test x.leaves == before_leaves
            @test x.cores == before_cores
            @test result.tree === x.tree
            result.leaves[first(tree.leaf_order)] .= 0
            @test x.leaves == before_leaves
            wrong = SumOfProducts([2], ProductTerm[])
            @test_throws DimensionMismatch apply(wrong, x)
            @test_throws DimensionMismatch dot(y, wrong, x)
        end
    end

    @testset "Single leaf, identity, zero and custom node IDs" begin
        x = product_state(balanced_tree(1), [ComplexF64[2im, 3]])
        op = SumOfProducts([2], [ProductTerm(2-im, Dict(1 => a)), ProductTerm(-1, Dict())])
        @test dense(op * x) ≈ ((2-im)*a - I) * dense(x)
        @test dot(x, op, x) ≈ dot(dense(x), ((2-im)*a-I)*dense(x))
        tree = BinaryTree([(2, 3), (0, 0), (0, 0)]; root=1, leaf_order=[3, 2])
        x = product_state(tree, [ComplexF64[2im, 3], [1, 2]])
        op = SumOfProducts([2, 2], [ProductTerm(1, Dict(1 => a))])
        @test vec(dense(op*x)) ≈ kron(Matrix{ComplexF64}(I, 2, 2), a) * vec(dense(x))
        for terms in (ProductTerm[], [ProductTerm(0, Dict(1 => a))])
            zero = SumOfProducts([2, 2], terms)
            @test iszero(dense(zero))
            @test iszero(dense(zero*x))
            @test iszero(dot(x, zero, x))
        end
        scalar = SumOfProducts([2, 2], [ProductTerm(2im, Dict())])
        @test dense(scalar*x) ≈ 2im * dense(x)
        @test dot(x, scalar, x) ≈ 2im * dot(x, x)
        xzero = product_state(tree, [zeros(2), ones(2)])
        @test iszero(dense(op*xzero))
        # Cancelling terms must not be normalized or approximated.
        cancel = SumOfProducts([2, 2], [ProductTerm(1, Dict(1 => a)), ProductTerm(-1, Dict(1 => a))])
        @test norm(dense(cancel*x)) < 1e-12
    end

    @testset "Ownership and validation" begin
        factor = copy(a)
        term = ProductTerm(1, Dict(1 => factor))
        factor .= 0
        @test term.factors[1] == a
        dims, terms = [2], [term]
        op = SumOfProducts(dims, terms)
        term.factors[1] .= 0
        empty!(terms)
        dims[1] = 9
        @test dense(op) == a
        result_dims = leafdims(op)
        result_dims[1] = 7
        @test leafdims(op) == [2]
        for coefficient in (Inf, NaN, BigFloat("1e400"))
            @test_throws ArgumentError ProductTerm(coefficient, Dict())
        end
        for factors in (Dict(0 => a), Dict(-1 => a), Dict(1.5 => a), Dict(1 => fill(NaN,2,2)))
            @test_throws ArgumentError ProductTerm(1, factors)
        end
        @test_throws DimensionMismatch ProductTerm(1, Dict(1 => ones(2,3)))
        @test_throws ArgumentError ProductTerm(1, Dict(1 => zeros(0,0)))
        @test_throws ArgumentError ProductTerm(1, Dict(1 => fill(BigFloat("1e400"),2,2)))
        for dimensions in (Int[], [0], [-1], [2.0])
            @test_throws ArgumentError SumOfProducts(dimensions, ProductTerm[])
        end
        @test_throws ArgumentError SumOfProducts([2], [ProductTerm(1, Dict(2=>a))])
        @test_throws DimensionMismatch SumOfProducts([3], [ProductTerm(1, Dict(1=>a))])
    end
end
