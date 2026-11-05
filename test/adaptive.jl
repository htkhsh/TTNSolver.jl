@testset "Stable TTN distance" begin
    tree, leaves, cores, _ = three_leaf_fixture(; order=[3, 1, 2])
    x = TTNTensor(tree, leaves, cores)
    y = copy(x)
    y.cores[tree.root][1] += 1e-8
    reference = norm(dense(x)-dense(y))
    @test ttn_distance(x, y) ≈ reference rtol=1e-6
    @test ttn_distance(x, x) < 1e-12*norm(x)
    for scale in (1e-170, 1e170)
        a = product_state(balanced_tree(1), [scale .* ComplexF64[1, im]])
        b = product_state(balanced_tree(1), [scale .* ComplexF64[1, 0]])
        @test ttn_distance(a, b)/scale ≈ 1
    end
    tree = balanced_tree(40)
    factors = [ComplexF64[1, 0] for _ in 1:40]
    x = product_state(tree, factors)
    factors[1][2] = 1e-8
    y = product_state(tree, factors)
    @test ttn_distance(x, y) ≈ 1e-8 atol=1e-14
    @test_throws ArgumentError ttn_distance(x, product_state(balanced_tree(1), [[1, 0]]))
end

function adaptive_error(f)
    try
        f()
        return nothing
    catch error
        return error
    end
end

@testset "Adaptive BUG driver" begin
    tree = balanced_tree(1)
    initial = product_state(tree, [ComplexF64[1, im]])
    A = ComplexF64[-0.2 0.3im; 0.4 -0.7]
    op = SumOfProducts([2], [ProductTerm(1, Dict(1=>A))])
    times = [0.03, 0.11, 0.37]
    result = adaptive_evolve(initial, op, (0.0, 0.5); dt=0.04,
                             dtmax=0.08, saveat=times, atol=1e-10, rtol=1e-9)
    @test result.times == [0.0; times; 0.5]
    @test result.stats.accepted > length(result.times)-1
    @test result.stats.rejected == 0
    @test all(d -> d.accepted && d.dt <= 0.08 + eps() && d.error_ratio <= 1, result.diagnostics)
    for (time, state) in zip(result.times, result.values)
        @test dense(state) ≈ exp(time*A)*dense(initial) atol=2e-11
    end
    @test dense(initial) == ComplexF64[1, im]
    result.values[1].leaves[tree.root] .= 0
    @test norm(result.final_state) > 0
    @test dense(initial) == ComplexF64[1, im]

    # Observers cannot mutate the live integrator state.
    observed = adaptive_evolve(initial, op, (0.0, 0.1); dt=0.04,
        observe=s -> (fill!(s.leaves[s.tree.root], 0); 0.0))
    @test observed.values == [0.0, 0.0]
    @test dense(observed.final_state) ≈ exp(0.1*A)*dense(initial) atol=1e-12
    zero_span = adaptive_evolve(initial, op, (0.2, 0.2); dt=0.1)
    @test zero_span.times == [0.2]
    @test zero_span.stats.accepted == 0
    @test adaptive_evolve(initial, op, (0., 0.1); dt=0.1, saveat=()).times == [0., 0.1]

    @testset "Rollback and bounded failures" begin
        t = balanced_tree(2)
        entangled = TTNTensor(t, Dict(n=>Matrix{ComplexF64}(I, 2, 2) for n in t.leaf_order),
                             Dict(t.root=>reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2)))
        before = dense(entangled)
        zero = SumOfProducts([2, 2], ProductTerm[])
        error = adaptive_error(() -> adaptive_evolve(entangled, zero, (0., 1.);
            dt=0.2, maxrank=1, maxreject=3, atol=1e-6, rtol=0))
        @test error isa AdaptiveIntegrationError
        if error isa AdaptiveIntegrationError
            @test error.reason == :maxreject
            @test error.last_reason == :truncation
            @test error.time == 0
            @test length(error.diagnostics) == 3
            @test all(d -> !d.accepted, error.diagnostics)
            @test dense(error.state) ≈ before
        end
        @test dense(entangled) == before
        error = adaptive_error(() -> adaptive_evolve(entangled, zero, (0., 1.);
            dt=0.2, dtmin=0.15, maxrank=1, atol=1e-6, rtol=0))
        @test error isa AdaptiveIntegrationError
        @test error.reason == :dtmin
        error = adaptive_error(() -> adaptive_evolve(initial, op, (0., 1.);
            dt=0.1, maxsteps=1))
        @test error isa AdaptiveIntegrationError
        @test error.reason == :maxsteps
        @test error.time == 0.1
        @test dense(error.state) ≈ exp(0.1*A)*dense(initial) atol=1e-12
        error = adaptive_error(() -> adaptive_evolve(initial, op, (0., 1.);
            dt=0.2, krylovdim=1, krylov_maxiter=1, krylov_tol=1e-14, maxreject=2))
        @test error isa AdaptiveIntegrationError
        @test error.last_reason == :local_solve
        @test error.time == 0
        @test all(d -> d.local_failure isa LocalSolveFailure, error.diagnostics)
        @test occursin("local_solve", sprint(showerror, error))
        a = 1e16
        error = adaptive_error(() -> adaptive_evolve(initial, op, (a, nextfloat(a)); dt=2.0))
        @test error isa AdaptiveIntegrationError
        @test error.reason == :time_resolution
        error = adaptive_error(() -> adaptive_evolve(initial, op, (a, a+8); dt=3., dtmax=3.))
        @test error isa AdaptiveIntegrationError
        @test error.reason == :time_resolution
        @test isempty(error.diagnostics)
        zero = SumOfProducts([2], ProductTerm[])
        unequal = adaptive_evolve(initial, zero, (a, a+6); dt=6., dtmax=6.)
        @test only(unequal.diagnostics).richardson_factor == 1.25
    end

    @testset "Recover after a local Arnoldi failure" begin
        A4 = ComplexF64[-1 1 0 0; 0 -2 1 0; 0 0 -3 1; 1 0 0 -4]
        x = product_state(balanced_tree(1), [[1,0,0,0]])
        L = SumOfProducts([4], [ProductTerm(1, Dict(1=>A4))])
        recovered = adaptive_evolve(x, L, (0.,0.05); dt=0.05,
            krylovdim=2, krylov_maxiter=1, krylov_tol=0.01,
            atol=1e-3, rtol=0, maxsteps=100)
        @test recovered.stats.rejected > 0
        @test any(d -> d.reason == :local_solve, recovered.diagnostics)
        @test last(recovered.diagnostics).accepted
        @test norm(dense(recovered.final_state)-exp(0.05A4)*dense(x)) < 1e-3
    end

    @testset "Input validation before any step" begin
        for kwargs in ((;dt=0), (;dt=NaN), (;dt=0.1,atol=-1), (;dt=0.1,atol=0,rtol=0),
                       (;dt=0.1,rtol=Inf), (;dt=0.1,truncation_fraction=1),
                       (;dt=0.1,dtmin=0.2), (;dt=0.1,dtmax=0),
                       (;dt=0.1,maxsteps=0), (;dt=0.1,maxreject=0),
                       (;dt=0.1,krylovdim=0))
            @test_throws ArgumentError adaptive_evolve(initial, op, (0., 0.); kwargs...)
        end
        for saveat in ([0.2, 0.1], [0.2, 0.2], [-1.0], [NaN], [1.1])
            @test_throws ArgumentError adaptive_evolve(initial, op, (0., 1.); dt=0.1, saveat)
        end
        @test_throws ArgumentError adaptive_evolve(initial, op, (1., 0.); dt=0.1)
        @test_throws ArgumentError adaptive_evolve(initial, op, (0., Inf); dt=0.1)
        @test_throws DimensionMismatch adaptive_evolve(initial,
            SumOfProducts([3], ProductTerm[]), (0., 0.); dt=0.1)
    end
end

@testset "Typed local solve failure" begin
    x = product_state(balanced_tree(1), [ComplexF64[1, im]])
    A = ComplexF64[-1 2im; 0.3 -0.4im]
    op = SumOfProducts([2], [ProductTerm(1, Dict(1=>A))])
    @test_logs begin
        @test_throws LocalSolveFailure augmented_bug_step(x, op, 5.0;
            krylovdim=1, krylov_maxiter=1, krylov_tol=1e-14)
    end
    huge = product_state(balanced_tree(1), [[1e308]])
    growth = SumOfProducts([1], [ProductTerm(1, Dict())])
    @test_throws LocalSolveFailure augmented_bug_step(huge, growth, 1.)
    error = adaptive_error(() -> adaptive_evolve(huge, growth, (0.,1.); dt=1., maxreject=2))
    @test error isa AdaptiveIntegrationError
    @test error.last_reason == :local_solve
    @test any(d -> !isnothing(d.local_failure) && d.local_failure.reason == :nonfinite, error.diagnostics)
end
