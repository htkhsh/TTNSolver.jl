function channel_test_operator(rng,dims;three_body=true,scalar=true)
    m = length(dims)
    rand_op(d) = randn(rng,ComplexF64,d,d)
    shared = [rand_op(d) for d in dims]          # equal factors reused across terms
    terms = ProductTerm[]
    for k in 1:m
        push!(terms,ProductTerm(randn(rng,ComplexF64),Dict(k=>rand_op(dims[k]))))
        push!(terms,ProductTerm(randn(rng,ComplexF64),Dict(k=>shared[k])))
    end
    for k in 2:m                                  # star couplings through mode 1
        push!(terms,ProductTerm(randn(rng,ComplexF64),Dict(1=>shared[1],k=>rand_op(dims[k]))))
        push!(terms,ProductTerm(randn(rng,ComplexF64),Dict(1=>rand_op(dims[1]),k=>shared[k])))
    end
    for k in 1:m-1
        push!(terms,ProductTerm(randn(rng,ComplexF64),Dict(k=>shared[k],k+1=>shared[k+1])))
    end
    three_body && m >= 3 && push!(terms,ProductTerm(randn(rng,ComplexF64),
        Dict(1=>rand_op(dims[1]),2=>shared[2],m=>rand_op(dims[m]))))
    scalar && push!(terms,ProductTerm(0.3-0.2im,Dict{Int,Matrix{ComplexF64}}()))
    push!(terms,ProductTerm(0,Dict(1=>rand_op(dims[1]))))   # dropped zero term
    return SumOfProducts(dims,terms)
end

function channel_test_state(rng,tree,dims,rank)
    ranks = zeros(Int,length(tree.children))
    leaves = Dict{Int,Matrix{ComplexF64}}()
    cores = Dict{Int,Array{ComplexF64,3}}()
    for (mode,node) in enumerate(tree.leaf_order)
        ranks[node] = min(rank,dims[mode])
        leaves[node] = randn(rng,ComplexF64,dims[mode],ranks[node])
    end
    for node in tree.postorder
        TTNSolver._isleaf(tree,node) && continue
        a,b = tree.children[node]
        ranks[node] = node == tree.root ? 1 : min(rank,ranks[a]*ranks[b])
        cores[node] = randn(rng,ComplexF64,ranks[node],ranks[a],ranks[b])
    end
    return orthogonalize!(TTNTensor(tree,leaves,cores))
end

# Visit every edge with the orthogonality center moved as in a serial sweep,
# returning for each non-root node the edge operator Σ D ⊗ U (coefficients
# included) together with the local actions applied to a fixed probe.
function channel_edge_operators(A,x0,probe_rng)
    x = copy(x0)
    tree = x.tree
    modes = Dict(node=>mode for (mode,node) in enumerate(tree.leaf_order))
    up,_ = TTNSolver._tamen_upward(A,x,x)
    down = [Matrix{ComplexF64}[] for _ in tree.children]
    down[tree.root] = TTNSolver._root_boundary(A)
    orthonormal = TTNSolver._assume_orthonormal(A)
    coefficients = A isa SumOfProducts ? [t.coefficient for t in A.terms] : ones(ComplexF64,length(up[tree.root]))
    edges = Dict{Int,Matrix{ComplexF64}}()
    actions = Dict{Int,Any}()
    function edge(node)
        cs = A isa SumOfProducts ? coefficients : ones(ComplexF64,length(down[node]))
        return sum(cs[k] .* kron(transpose(up[node][k]),down[node][k]) for k in eachindex(down[node]))
    end
    function probe!(node)
        if TTNSolver._isleaf(tree,node)
            X = x.leaves[node]
            act = TTNSolver._leaf_action(A,node,modes[node],down[node],X)
        else
            l,r = tree.children[node]
            X = x.cores[node]
            act = TTNSolver._core_action(A,node,down[node],up[l],up[r],X)
        end
        actions[node] = act(X)
    end
    function visit!(node)
        probe!(node)
        TTNSolver._isleaf(tree,node) && return
        left,right = tree.children[node]
        for (child,sibling,mode) in ((left,right,2),(right,left,3))
            Q = TTNSolver._tamen_down!(x,node,child,mode)
            down[child] = TTNSolver._env_down(A,node,mode,Q,down[node],up[sibling];orthonormal)
            edges[child] = edge(child)
            visit!(child)
            TTNSolver._tamen_up!(x,node,child,mode)
            if TTNSolver._isleaf(tree,child)
                up[child] = TTNSolver._env_leaf(A,child,modes[child],x.leaves[child];orthonormal)
            else
                a,c = tree.children[child]
                up[child] = TTNSolver._env_up(A,child,TTNSolver._basis_matrix(x.cores[child]),
                    x.cores[child],up[a],up[c];orthonormal)
            end
        end
    end
    visit!(tree.root)
    return edges,actions
end

function toy_split_heom(sites;bath=3)
    # Split ket/bra system leaves coupled to two modes per site, as in HEOM.
    d = sites
    H = randn(MersenneTwister(1),ComplexF64,d,d); H = H+H'
    terms = [ProductTerm(-im,Dict(1=>H)),ProductTerm(im,Dict(2=>transpose(H)))]
    lower = diagm(1=>sqrt.(ComplexF64.(1:bath-1)))
    number = diagm(0=>ComplexF64.(0:bath-1))
    for j in 1:sites, k in 1:2
        V = zeros(ComplexF64,d,d); V[j,j] = 1
        m = 2 + 2(j-1) + k
        push!(terms,ProductTerm(-0.5-0.1k,Dict(m=>number)))
        push!(terms,ProductTerm(-0.7im,Dict(1=>V,m=>lower)))
        push!(terms,ProductTerm(0.7im,Dict(2=>V,m=>lower)))
        push!(terms,ProductTerm(-0.3im-0.1,Dict(1=>V,m=>lower')))
        push!(terms,ProductTerm(0.3im-0.1,Dict(2=>V,m=>lower')))
    end
    return SumOfProducts([d;d;fill(bath,2sites)],terms)
end

@testset "Exact channel environments" begin
    rng = MersenneTwister(2024)
    trees = [balanced_tree(5),
             BinaryTree([(2,3),(0,0),(4,5),(0,0),(6,7),(0,0),(0,0)]; root=1,leaf_order=[6,2,7,4]),
             BinaryTree([(0,0),(0,0),(1,2),(0,0),(3,4),(0,0),(5,6)])]
    dimsets = [[2,3,2,3,2],[3,2,2,3],[2,2,3,2]]
    for (tree,dims) in zip(trees,dimsets)
        op = channel_test_operator(rng,dims)
        P = TTNSolver._channel_operator(op,tree)
        x = channel_test_state(rng,tree,dims,4)
        # Projected root operator gives the full expectation value.
        up,_ = TTNSolver._tamen_upward(P,x,x)
        root = only(up[tree.root])
        @test only(root) ≈ dot(x,op,x) rtol=1e-10
        es,as = channel_edge_operators(op,x,rng)
        ec,ac = channel_edge_operators(P,x,rng)
        for node in keys(es)
            @test ec[node] ≈ es[node] rtol=1e-10 atol=1e-11*norm(es[node])
        end
        for node in keys(as)
            @test ac[node] ≈ as[node] rtol=1e-10 atol=1e-11*norm(as[node])
        end
    end
    let dims = [2,2,2], op = channel_test_operator(rng,dims;three_body=false)
        P = TTNSolver._channel_operator(op,balanced_tree(3))
        other = BinaryTree([(0,0),(0,0),(0,0),(2,3),(1,4)])
        @test other.children != balanced_tree(3).children
        @test_throws DimensionMismatch TTNSolver._check_operator(P,channel_test_state(rng,other,dims,2))
        @test_throws DimensionMismatch TTNSolver._channel_operator(op,balanced_tree(4))
    end

    @testset "HEOM channel counts" begin
        sites = 4
        op = toy_split_heom(sites)
        modes = length(op.dimensions)
        system = BinaryTree([(0,0),(0,0),(1,2)])
        bath = balanced_tree(modes-2)
        offset = 3
        children = [system.children; [(l==0 ? 0 : l+offset, r==0 ? 0 : r+offset) for (l,r) in bath.children]]
        push!(children,(system.root,bath.root+offset))
        tree = BinaryTree(children)
        P = TTNSolver._channel_operator(op,tree)
        # The bath root carries identity, completed terms and 2 per site.
        @test P.channels[bath.root+offset] == 2+2sites
        # System edges carry identity, completed terms and one bare V per site/side.
        @test P.channels[system.root] == 2+2sites
        @test P.channels[1] == 2+sites
        # Every bath leaf: identity, number operator and ket/bra coupling sums.
        @test all(P.channels[node] == 4 for node in tree.leaf_order[3:end])
        x = channel_test_state(rng,tree,op.dimensions,3)
        es,_ = channel_edge_operators(op,x,rng)
        ec,_ = channel_edge_operators(P,x,rng)
        @test all(ec[n] ≈ es[n] for n in keys(es))
    end

    @testset "tamen_step agreement" begin
        op,x = tt_acceptance_model(seed=8)
        exact = tt_exact(op,x,0.5)
        for als_sweep in (:serial,:etter), threaded in (false,true), certify in (true,false)
            kw = (;time_points=8,rtol=1e-9,maxrank=9,maxsweeps=20,kickrank=2,als_sweep,threaded,
                  certify,estimate_time_error=true)
            ys,is = tamen_step(x,op,0.5;kw...)
            yc,ic = tamen_step(x,op,0.5;kw...,local_operator=:channels)
            @test ic.converged && ic.certified == certify
            @test tt_error(yc,exact) < 1e-7
            @test ttn_distance(yc,ys) <= 1e-7*norm(ys)
            @test ic.time_error.absolute ≈ is.time_error.absolute rtol=1e-4 atol=1e-12
            @test ic.time_error.initial_projection_error ≈ is.time_error.initial_projection_error rtol=1e-6 atol=1e-13
        end
        @test_throws ArgumentError tamen_step(x,op,0.5;local_operator=:dense)
    end

    @testset "split HEOM-like operator and adaptive intervals" begin
        op = toy_split_heom(2;bath=2)
        tree = balanced_tree(length(op.dimensions))
        rho = ComplexF64[0.7 0.2im; -0.2im 0.3]
        F = svd(rho)
        # Rank-1 split system state with vacuum bath modes.
        vac = ComplexF64[1,0]
        x = product_state(tree,[F.U[:,1]*sqrt(F.S[1]),conj(F.V[:,1])*sqrt(F.S[1]),vac,vac,vac,vac])
        kw = (;time_points=6,rtol=1e-8,maxrank=8,maxsweeps=20,kickrank=2,
              certify=false,truncation_rtol=1e-10,local_tol_factor=0.01)
        ys,_ = tamen_step(x,op,0.3;kw...)
        yc,ic = tamen_step(x,op,0.3;kw...,local_operator=:channels)
        @test ic.converged
        exact = exp(0.3dense(op))*vec(dense(x))
        # Six Chebyshev points leave a small time-discretization error in both.
        sop_error = norm(vec(dense(ys))-exact)
        @test norm(vec(dense(yc))-exact) <= 1.05sop_error + 1e-9
        @test sop_error < 1e-5*norm(dense(x))
        @test ttn_distance(yc,ys) <= 1e-6*norm(ys)
        result = tamen_evolve(x,op,(0.0,0.3);dt=0.15,rtol=1e-6,atol=1e-9,solve_rtol=1e-8,
            maxrank=8,kickrank=2,local_operator=:channels,certify=false,local_tol_factor=0.01)
        @test result.accepted >= 2
        @test norm(vec(dense(result.final_state))-exp(0.3dense(op))*vec(dense(x))) < 1e-5*norm(dense(x))
    end
end

@testset "Channel environments for a zero operator" begin
    tree = balanced_tree(2)
    x = product_state(tree,[ComplexF64[1,0.2im,0.1],ComplexF64[1,0.3im]])
    zeroop = SumOfProducts([3,2],ProductTerm[])
    for lo in (:sop,:channels)
        y,z = tamen_step(x,zeroop,0.2;estimate_time_error=true,local_operator=lo)
        @test z.converged
        @test dense(y) ≈ dense(x)
        @test z.time_error.absolute < 1e-12
    end
end
