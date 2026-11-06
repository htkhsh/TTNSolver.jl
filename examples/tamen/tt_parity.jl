# Optional sibling-package check, from the .julia/dev parent directory:
# JULIA_LOAD_PATH="$(pwd)/TTNSolver:$(pwd)/TTSolver:@stdlib" julia TTNSolver/examples/tamen/tt_parity.jl
# TTSolver is needed only for this example, not by TTNSolver itself.
using LinearAlgebra, Random, Test
import TTNSolver as N
import TTSolver as T

BLAS.set_num_threads(1)
L = [-0.2 0.1;0.2 -0.1]
x0 = [0.8,0.2]
xn = N.product_state(N.balanced_tree(1),[x0])
Ln = N.SumOfProducts([2],[N.ProductTerm(1,Dict(1=>L))])

@testset "TT / TTN fixed-interval tAMEn parity" begin
    for scheme in ("cheb","CN")
        Random.seed!(20261106)
        xt = T.TTTensor([reshape(copy(x0),1,2,1)])
        spacetime = T.tkron(xt,T.tt_ones(5))
        Lt = T.TTMatrix([reshape(copy(L),1,2,2,1)])
        _,_,_,yt,_ = T.tamen(spacetime,Lt,1e-10;nswp=8,kickrank=2,verb=0,
            local_iters=30,time_scheme=scheme,time_error_damp=0.,correct_2_norm=false)
        yn,d = N.tamen_step(xn,Ln,1.;time_points=5,time_scheme=scheme,rtol=1e-11)
        difference = norm(vec(N.dense(yn))-vec(T.tt_full(yt)))
        @test difference < 1e-10
        @test d.converged
        println((;scheme,difference,ttn_error=norm(vec(N.dense(yn))-exp(L)*x0)))
    end
end
