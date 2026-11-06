include("fixtures/tamen_canonical_reference.jl")

@testset "Residual scratch lifetime and rank changes" begin
    rng=MersenneTwister(6432)
    work=TTNSolver._ResidualWorkspace()
    for tree in (balanced_tree(1),balanced_tree(4),
                 BinaryTree([(2,3),(0,0),(4,5),(0,0),(0,0)];root=1,leaf_order=[5,2,4]))
        d=length(tree.leaf_order)
        for rank in (2,6,1,6)
            x=certification_state(tree,fill(3,d),fill(rank,length(tree.children)),rng)
            y=certification_state(tree,fill(3,d),fill(2,length(tree.children)),rng)
            term=ProductTerm(0.2-0.3im,Dict(d=>randn(rng,ComplexF64,3,3)))
            expected=vec(dense(x))-dense(SumOfProducts(fill(3,d),[term]))*vec(dense(y))
            z=TTNSolver._tamen_add_action_canonical(x,y,term,-1;workspace=work)
            @test vec(dense(z)) ≈ expected rtol=2e-12
            snapshot=dense(z)
            counts=TTNSolver._residual_work_counts(work)
            again=TTNSolver._tamen_add_action_canonical(x,y,term,-1;workspace=work)
            after=TTNSolver._residual_work_counts(work)
            @test after.residual_buffer_grows==counts.residual_buffer_grows
            @test after.residual_buffer_reuses>counts.residual_buffer_reuses
            @test dense(again) ≈ snapshot rtol=1e-12
            # Poison all retained scratch: outputs must be owned and all active
            # input regions must be filled on the next use, including shrinkage.
            for node in work.nodes, storage in node.storage; fill!(storage,NaN); end
            @test dense(z)==snapshot
            clean=TTNSolver._tamen_add_action_canonical(x,y,term,-1;workspace=work)
            @test vec(dense(clean)) ≈ expected rtol=2e-12
            @test dense(z)==snapshot
        end
    end
end

@testset "Parallel canonical sums and failure joining" begin
    rng=MersenneTwister(345)
    tree=balanced_tree(8)
    x=certification_state(tree,fill(4,8),fill(3,length(tree.children)),rng)
    y=certification_state(tree,fill(4,8),fill(2,length(tree.children)),rng)
    term=ProductTerm(0.3im,Dict(1=>randn(rng,ComplexF64,4,4),8=>randn(rng,ComplexF64,4,4)))
    original_x,original_y=dense(x),dense(y)
    serial=TTNSolver._tamen_add_action_canonical(x,y,term,-1)
    work=TTNSolver._ResidualWorkspace()
    parallel=TTNSolver._tamen_add_action_canonical(x,y,term,-1;workspace=work,threaded=true)
    @test ttn_distance(serial,parallel)<1e-11*norm(serial)
    @test (TTNSolver._residual_work_counts(work).residual_tasks>0)==(Threads.nthreads(:default)>1)
    @test dense(x)==original_x && dense(y)==original_y
    # A bad factor fails inside a child, after sibling work may have started.
    @test_throws DimensionMismatch TTNSolver._tamen_add_action_canonical(x,y,
        ProductTerm(1,Dict(1=>ones(2,2))),-1;workspace=work,threaded=true)
    @test !work.busy[]
    repaired=TTNSolver._tamen_add_action_canonical(x,y,term,-1;workspace=work,threaded=true)
    @test ttn_distance(serial,repaired)<1e-11*norm(serial)
    TTNSolver._with_residual_workspace(work) do w
        @test_throws ArgumentError TTNSolver._tamen_add_action_canonical(x,y,term,-1;workspace=w,threaded=true)
    end
    tasks=[Threads.@spawn TTNSolver._tamen_add_action_canonical(x,y,term,-1;
        workspace=TTNSolver._ResidualWorkspace(),threaded=true) for _ in 1:2]
    a,b=fetch.(tasks)
    @test ttn_distance(a,b)<1e-11*norm(a)
end

@testset "Buffered threaded certificate and public workspace" begin
    rng=MersenneTwister(399)
    tree=balanced_tree(4)
    x=certification_state(tree,fill(3,4),fill(2,length(tree.children)),rng)
    b=certification_state(tree,fill(3,4),fill(3,length(tree.children)),rng)
    A=SumOfProducts(fill(3,4),[ProductTerm(0.2im,Dict(j=>randn(rng,ComplexF64,3,3))) for j in 1:4])
    exact=vec(dense(b))-dense(A)*vec(dense(x))
    work=TTNSolver._ResidualWorkspace()
    for cap in (1,5,20)
        ref=reference_canonical_residual(A,x,b,1e-10,cap)
        for reuse in (false,true), threaded in (false,true)
            z,bound,drop=TTNSolver._tamen_residual(A,x,b,1e-10,cap;workspace=reuse ? work : nothing,threaded)
            @test norm(exact)<=bound+1e-10*norm(exact)
            @test norm(vec(dense(z))-exact)<=drop+1e-10*norm(exact)
            @test ttn_distance(z,ref[1])<=drop+ref[3]+1e-10*norm(exact)
            @test abs(bound-ref[2])<=2(drop+ref[3])+1e-10*norm(exact)
        end
    end
    initial=product_state(balanced_tree(3),[[1,0],[1,0],[1,0]])
    sx=ComplexF64[0 1;1 0]
    L=SumOfProducts([2,2,2],[ProductTerm(-0.3im,Dict(1=>sx,2=>sx)),ProductTerm(-0.1im,Dict(2=>sx,3=>sx))])
    workspace=TAMEnWorkspace()
    y,d=tamen_step(initial,L,0.3;time_points=5,rtol=1e-9,workspace,threaded=true,als_sweep=:etter)
    @test d.converged && d.relative_residual_bound<=1e-9
    @test d.workspace_stats.residual_buffer_grows>0
    @test d.workspace_stats.residual_buffer_reuses>0
    @test (d.workspace_stats.residual_tasks>0)==(Threads.nthreads(:default)>1)
    @test norm(vec(dense(y))-exp(0.3dense(L))*vec(dense(initial)))<1e-8
end
