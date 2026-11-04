function test_root_gauge(state)
    for node in eachindex(state.tree.children)
        node == state.tree.root && continue
        basis = if state.tree.children[node] == (0, 0)
            state.leaves[node]
        else
            core = state.cores[node]
            transpose(reshape(core, size(core, 1), :))
        end
        @test basis' * basis ≈ I atol=1e-12 rtol=1e-12
    end
end

@testset "TTN algebra" begin
    @testset "Conjugate inner products with unequal ranks" begin
        tree, leaves, cores, reference = three_leaf_fixture()
        x = TTNTensor(tree, leaves, cores)
        other_tree, other_leaves, other_cores, other_reference =
            three_leaf_fixture(; ranks=(1, 3, 1, 2), seed=955)
        y = TTNTensor(other_tree, other_leaves, other_cores)
        @test dot(x, y) ≈ dot(vec(reference), vec(other_reference)) atol=1e-12
        @test dot(y, x) ≈ conj(dot(x, y)) atol=1e-12
        @test norm(x) ≈ norm(reference) atol=1e-12
        @test dot(x, x) ≈ norm(reference)^2 atol=1e-12
        wrong_dims = product_state(tree, [ones(3), ones(3), ones(2)])
        @test_throws DimensionMismatch dot(x, wrong_dims)
        other_order = TTNTensor(BinaryTree(tree.children; leaf_order=[3, 1, 2]), leaves, cores)
        @test_throws ArgumentError dot(x, other_order)
        other_topology = product_state(balanced_tree(3), [ones(2), ones(3), ones(2)])
        @test_throws ArgumentError dot(x, other_topology)
        @test dot(x, y) != dot(vec(conj(reference)), vec(other_reference))
    end

    @testset "QR preserves tensor, amplitude and root gauge" begin
        for ranks in ((2, 2, 2, 3), (4, 4, 3, 15))
            tree, leaves, cores, reference = three_leaf_fixture(; ranks)
            x = TTNTensor(tree, leaves, cores)
            before = copy(x)
            @test orthogonalize!(x) === x
            @test dense(x) ≈ reference atol=1e-12 rtol=1e-12
            @test norm(x) ≈ norm(reference) atol=1e-12 rtol=1e-12
            @test norm(x.cores[tree.root]) ≈ norm(reference) atol=1e-12 rtol=1e-12
            @test dense(before) ≈ reference atol=1e-12 rtol=1e-12
            test_root_gauge(x)
            orthogonalize!(x)
            @test dense(x) ≈ reference atol=1e-12 rtol=1e-12
            test_root_gauge(x)
            if ranks[1] > 2
                @test tree_ranks(x) == [2, 3, 2, 6, 1]
            end
        end

        tree, leaves, cores, _ = three_leaf_fixture()
        leaves[1][:, 2] = leaves[1][:, 1]
        cores[4][2, :, :] = cores[4][1, :, :]
        deficient = TTNTensor(tree, leaves, cores)
        reference = dense(deficient)
        orthogonalize!(deficient)
        @test dense(deficient) ≈ reference atol=1e-12 rtol=1e-12
        test_root_gauge(deficient)
    end

    @testset "Arbitrary node numbering and leaf order" begin
        _, leaves, cores, reference = three_leaf_fixture()
        tree = BinaryTree([(2, 3), (0, 0), (4, 5), (0, 0), (0, 0)];
                          root=1, leaf_order=[5, 2, 4])
        x = TTNTensor(tree, Dict(2 => leaves[1], 4 => leaves[2], 5 => leaves[3]),
                      Dict(1 => cores[5], 3 => cores[4]))
        @test dense(x) ≈ permutedims(reference, (3, 1, 2)) atol=1e-12
        orthogonalize!(x)
        @test dense(x) ≈ permutedims(reference, (3, 1, 2)) atol=1e-12
        @test norm(x) ≈ norm(reference) atol=1e-12
        test_root_gauge(x)
    end

    @testset "One leaf, zeros and many-leaf products" begin
        single = product_state(balanced_tree(1), [ComplexF64[2im, 3]])
        orthogonalize!(single)
        @test dense(single) == ComplexF64[2im, 3]
        @test norm(single) ≈ sqrt(13)
        @test dot(single, single) ≈ 13
        for count in (1, 5)
            zero = product_state(balanced_tree(count), [zeros(2) for _ in 1:count])
            orthogonalize!(zero)
            @test iszero(dense(zero))
            @test iszero(norm(zero))
            test_root_gauge(zero)
        end
        many = product_state(balanced_tree(12), [ComplexF64[1, 2im] for _ in 1:12])
        @test dot(many, many) ≈ 5.0^12
        orthogonalize!(many)
        @test norm(many) ≈ 5.0^6
        test_root_gauge(many)
    end
end
