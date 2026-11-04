@testset "Binary tree topology" begin
    @testset "Balanced traversal and parent relationships" begin
        for n in (1, 3, 5)
            tree = balanced_tree(n)
            @test length(tree.children) == 2n - 1
            @test tree.leaf_order == collect(1:n)
            @test tree.parent[tree.root] == 0
            @test sort(tree.postorder) == collect(eachindex(tree.children))
            positions = invperm(tree.postorder)
            for (node, children) in pairs(tree.children)
                children == (0, 0) && continue
                for child in children
                    @test tree.parent[child] == node
                    @test positions[child] < positions[node]
                end
            end
        end
        @test_throws ArgumentError balanced_tree(0)
        @test_throws ArgumentError balanced_tree(-1)
    end

    @testset "Explicit order and owned inputs" begin
        children = [(2, 3), (0, 0), (0, 0)]
        order = [3, 2]
        tree = BinaryTree(children; root=1, leaf_order=order)
        children[1] = (0, 0)
        order[1] = 2
        @test tree.children[1] == (2, 3)
        @test tree.leaf_order == [3, 2]
        @test tree.postorder == [2, 3, 1]
        @test tree.parent == [0, 1, 1]
    end

    @testset "Reject malformed topology" begin
        @test_throws ArgumentError BinaryTree(Tuple{Int,Int}[])
        @test_throws ArgumentError BinaryTree([(0, 0)]; root=0)
        @test_throws ArgumentError BinaryTree([(0, 0)]; root=2)
        @test_throws ArgumentError BinaryTree([(0, 0), (1, 0)])
        @test_throws ArgumentError BinaryTree([(0, 0), (1, 3)])
        @test_throws ArgumentError BinaryTree([(0, 0), (-1, 1)])
        @test_throws ArgumentError BinaryTree([(0, 0), (1, 1)])
        @test_throws ArgumentError BinaryTree([(0, 0), (0, 0)])
        @test_throws ArgumentError BinaryTree([(2, 3), (1, 3), (0, 0)]; root=1)
        # A disconnected cycle plus a valid leaf root.
        @test_throws ArgumentError BinaryTree([(2, 4), (1, 5), (0, 0), (0, 0), (0, 0)]; root=3)
        @test_throws ArgumentError BinaryTree([(0, 0, 0)])
        @test_throws ArgumentError BinaryTree([(0.0, 0.0)])
        children = [(0, 0), (0, 0), (1, 2)]
        for order in ([1], [1, 1], [1, 3], [1, 2, 3], [1.0, 2.0])
            @test_throws ArgumentError BinaryTree(children; leaf_order=order)
        end
    end
end
