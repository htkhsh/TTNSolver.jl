@testset "Tree passes over parallel subtrees" begin
    rng = MersenneTwister(91)
    function chain_tree(n)
        children = [(0,0) for _ in 1:n]
        previous = 1
        for k in 2:n
            push!(children,(previous,k))
            previous = length(children)
        end
        return BinaryTree(children; root=length(children))
    end
    function random_ttn(tree,d,r)
        leaves = Dict(n=>randn(rng,ComplexF64,d,r) for n in tree.leaf_order)
        cores = Dict(n=>randn(rng,ComplexF64,n==tree.root ? 1 : r,r,r)
                     for n in eachindex(tree.children) if tree.children[n] != (0,0))
        return TTNTensor(tree,leaves,cores)
    end
    same(a,b) = a.leaves == b.leaves && a.cores == b.cores
    for tree in (balanced_tree(8),chain_tree(6),balanced_tree(5)), r in (2,3)
        x = random_ttn(tree,4,r)
        c = product_state(tree,[randn(rng,ComplexF64,4) for _ in tree.leaf_order])
        serial = orthogonalize!(copy(x))
        threaded = orthogonalize!(copy(x);threaded=true)
        @test same(serial,threaded)
        @test dense(threaded) ≈ dense(x)
        e = enrich_basis(x,c)
        @test same(e,enrich_basis(x,c;threaded=true))
        @test dense(e) ≈ dense(x)
        for weighting in (:weighted,:local), fixed_leaves in (Int[],[first(tree.leaf_order)])
            a,b = copy(e),copy(e)
            kw = (;atol=0.05*norm(x),maxrank=2,conserved=c,weighting,fixed_leaves)
            ia = truncate!(a;kw...)
            ib = truncate!(b;kw...,threaded=true)
            @test same(a,b)
            @test ia == ib
            @test dot(c,b) ≈ dot(c,x) rtol=1e-10
            @test norm(dense(b)-dense(x)) <= ib.discarded_bound*(1+1e-10) + 1e-12
        end
    end
end
