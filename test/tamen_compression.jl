@testset "Global residual phase timing and workspace deltas" begin
    x=product_state(balanced_tree(3),[[1,0],[1,0],[1,0]])
    sx=ComplexF64[0 1;1 0]
    A=SumOfProducts([2,2,2],[ProductTerm(-0.2im,Dict(1=>sx,2=>sx)),
                             ProductTerm(-0.3im,Dict(2=>sx,3=>sx))])
    work=TAMEnWorkspace()
    _,first=tamen_step(x,A,0.2;workspace=work,threaded=true,time_points=5,rtol=1e-9)
    @test hasproperty(first.timings,:residual_global_parts)
    if hasproperty(first.timings,:residual_global_parts)
        before=TTNSolver._residual_times(work.certification)
        _,second=tamen_step(x,A,0.2;workspace=work,threaded=true,time_points=5,rtol=1e-9)
        after=TTNSolver._residual_times(work.certification)
        parts=second.timings.residual_global_parts
        @test keys(parts)==(:addition_seconds,:weight_qr_seconds,:svd_seconds,:basis_update_seconds,:final_qr_seconds)
        @test all(>=(0),values(parts))
        @test all(>(0),values(parts))
        @test values(parts)==values(after).-values(before)
        @test sum(values(parts))<=second.timings.residual_global_seconds*1.001
        @test second.converged && first.converged
    end
end

@testset "Weighted cut QR block boundaries" begin
    rng=MersenneTwister(812)
    work=TTNSolver._ResidualCutWorkspace()
    for (p,l,r) in ((3,35,14),(3,36,14),(3,37,14),(3,14,37),(1,80,38)), mode in (2,3)
        C=randn(rng,ComplexF64,p,l,r)
        F=randn(rng,ComplexF64,3,p)
        A=TTNSolver._complement_matrix(TTNSolver._parent_product(C,F),mode)
        R,M=TTNSolver._cut_weight!(work,C,F,mode)
        @test R'*R≈A'*A rtol=1e-12
        expected=copy(M)
        oldW,oldtail=TTNSolver._retained_space(expected,zeros(ComplexF64,size(M,1),0),1e-9,17)
        W,tail=TTNSolver._cut_space!(M,1e-9,17)
        @test tail≈oldtail atol=1e-10 rtol=1e-12
        @test W*W'≈oldW*oldW' atol=1e-10
        @test norm(expected-W*(W'*expected))≈tail atol=1e-10 rtol=1e-12
    end
end

@testset "Reusable unprotected weighted cuts" begin
    @test isdefined(TTNSolver,:_ResidualCutWorkspace)
    if isdefined(TTNSolver,:_ResidualCutWorkspace)
        rng=MersenneTwister(2983)
        work=TTNSolver._ResidualCutWorkspace()
        trees=(balanced_tree(1),balanced_tree(4),
            BinaryTree([(2,3),(0,0),(4,5),(0,0),(0,0)];root=1,leaf_order=[5,2,4]))
        for tree in trees, rank in (2,7,1,7), cap in (1,4,30), tol in (0.0,1e-8)
            dims=fill(3,length(tree.leaf_order))
            original=orthogonalize!(certification_state(tree,dims,fill(rank,length(tree.children)),rng))
            root=tree.root
            rootdata=TTNSolver._isleaf(tree,root) ? original.leaves[root] : original.cores[root]
            rootdata./=norm(original)
            x,reference=copy(original),copy(original)
            old=TTNSolver._truncate_canonical!(reference;atol=tol,maxrank=cap,canonicalize=false)
            result=TTNSolver._residual_truncate!(x,work;atol=tol,maxrank=cap)
            @test maximum(tree_ranks(x))<=cap
            @test norm(vec(dense(x))-vec(dense(original)))<=result.discarded_bound+1e-11
            @test result.discarded_bound≈old.discarded_bound atol=1e-11 rtol=1e-10
            @test norm(vec(dense(x))-vec(dense(reference)))<1e-10
            snapshot=dense(x)
            counts=work.grows
            TTNSolver._residual_truncate!(copy(original),work;atol=tol,maxrank=cap)
            @test work.grows==counts
            for buffer in (work.storage...,work.factors...);fill!(buffer,NaN);end
            @test dense(x)==snapshot
            clean=copy(original)
            TTNSolver._residual_truncate!(clean,work;atol=tol,maxrank=cap)
            @test norm(vec(dense(clean))-vec(snapshot))<1e-10
        end
        for scale in (0.0,1e-100), cap in (1,20)
            x=orthogonalize!(certification_state(balanced_tree(4),fill(3,4),fill(4,7),rng))
            x.cores[x.tree.root].*=scale
            before=copy(x)
            result=TTNSolver._residual_truncate!(x,work;atol=1e-110,maxrank=cap)
            @test norm(vec(dense(x))-vec(dense(before)))<=result.discarded_bound+1e-12*norm(before)
            @test all(isfinite,dense(x))
        end
    end
end
