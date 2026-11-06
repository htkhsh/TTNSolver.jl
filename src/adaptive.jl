"""Adaptive integration stopped at a limit. `state` is the last accepted TTN;
`time`, `last_reason` and the complete attempt `diagnostics` support inspection.
"""
struct AdaptiveIntegrationError <: Exception
    reason::Symbol
    time::Float64
    dt::Float64
    last_reason::Symbol
    state::TTNTensor
    diagnostics::Vector{NamedTuple}
end

function Base.showerror(io::IO, error::AdaptiveIntegrationError)
    print(io, "adaptive BUG stopped: ", error.reason, " at t=", error.time,
          ", dt=", error.dt, "; last attempt: ", error.last_reason)
    error.last_reason == :truncation && print(io, "; increase maxrank or relax the truncation budget")
    error.last_reason == :local_solve && print(io, "; inspect local solver tolerances and Krylov limits")
end

function _adaptive_stop(reason, time, dt, state, history)
    last_reason = isempty(history) ? :none : last(history).reason
    throw(AdaptiveIntegrationError(reason, time, dt, last_reason, copy(state), history))
end

function _nonnegative_float(value, name)
    value isa Real && isfinite(value) && value >= 0 ||
        throw(ArgumentError("$name must be finite, real and nonnegative"))
    result = Float64(value)
    isfinite(result) || throw(ArgumentError("$name must be finite in Float64"))
    return result
end

function _save_grid(tspan, saveat)
    length(tspan) == 2 && all(t -> t isa Real && isfinite(t), tspan) ||
        throw(ArgumentError("tspan must contain two finite real times"))
    start, finish = Float64.(tspan)
    isfinite(start) && isfinite(finish) && isfinite(finish-start) && start <= finish ||
        throw(ArgumentError("tspan must be finite and ordered in Float64"))
    if isnothing(saveat)
        return start == finish ? [start] : [start, finish]
    end
    saveat isa Union{AbstractVector,Tuple} ||
        throw(ArgumentError("saveat must be an increasing collection of times"))
    all(t -> t isa Real && isfinite(t), saveat) || throw(ArgumentError("saveat must be finite and real"))
    grid = Float64[t for t in saveat]
    all(t -> start <= t <= finish, grid) && all(>(0), diff(grid)) ||
        throw(ArgumentError("saveat must be strictly increasing and inside tspan"))
    (isempty(grid) || first(grid) != start) && pushfirst!(grid, start)
    last(grid) != finish && push!(grid, finish)
    return grid
end

# Reject immediately if any branch's reference is insufficiently resolved.
# Only LocalSolveFailure is recoverable; programming/argument errors propagate.
function _adaptive_trial(state, operator, h1, h2, budget, options, timer=nothing;
                         method=:augmented, threaded=true)
    coarse, half1, half2 = nothing, nothing, nothing
    stage = :coarse
    finish(candidate, reason, failure=nothing) =
        (; candidate, reason, stage, local_failure=failure, coarse, half1, half2)
    try
        prepare = method == :parallel ? _prepare_parallel_bug_step : _prepare_bug_step
        prepared_step = method == :parallel ? _prepared_parallel_bug_step : _prepared_bug_step
        step = method == :parallel ? parallel_bug_step : augmented_bug_step
        execution = (;threaded)
        step_options = merge(Base.structdiff(options,(;conserved=options.conserved)),execution)
        # These two branches have the same old state. Charge preparation once
        # to the coarse solve; the second half must prepare its changed input.
        full, coarse, prepared = _measure_adaptive(timer,Val(:coarse_bug)) do
            prepared = prepare(state,operator,options.conserved,options.maxrank)
            full,info = prepared_step(prepared,operator,h1+h2;atol=budget,step_options...)
            (full,info,prepared)
        end
        coarse.truncation.tolerance_met || return finish(nothing, :truncation)
        stage = :half1
        middle, half1 = _measure_adaptive(timer,Val(:fine_bug)) do
            prepared_step(prepared,operator,h1;atol=budget/2,step_options...)
        end
        half1.truncation.tolerance_met || return finish(nothing, :truncation)
        stage = :half2
        fine, half2 = _measure_adaptive(timer,Val(:fine_bug)) do
            step(middle, operator, h2; atol=budget/2, options..., execution...)
        end
        half2.truncation.tolerance_met || return finish(nothing, :truncation)
        stage = :complete
        difference = _measure_adaptive(timer,Val(:difference)) do
            ttn_distance(full,fine)
        end
        return merge(finish(fine, :candidate), (; difference))
    catch error
        error isa LocalSolveFailure || rethrow()
        return finish(nothing, :local_solve, error)
    end
end

"""
    adaptive_evolve(initial, operator, tspan; dt, saveat=nothing,
        atol=1e-6, rtol=1e-4, truncation_fraction=0.1, dtmin=0, dtmax=Inf,
        maxsteps=100000, maxreject=20, observe=copy, maxrank=typemax(Int),
        conserved=nothing, krylov_tol=1e-12, krylovdim=30, krylov_maxiter=100,
        correct_invariant=true, record_timing=false, method=:augmented, threaded=true,
        truncation=:weighted)

Adaptive first-order BUG for an autonomous linear generator. Select
`method=:augmented` (default) or `:parallel`. `threaded` controls concurrent
subtree updates for augmented BUG and all-node solves for parallel BUG.
`truncation=:weighted` uses amplitude-weighted compression by default;
`:local` selects conservative local-core compression for all three trial steps.
This controller uses step doubling, not
the parallel paper's optional rank-expansion retry strategy. Compare
one full step with two half steps and accept the two-half-step result. Error
estimates use the whole TTN and include coarse truncation/correction norms plus
twice those of the fine branch for equal halves. For representably unequal
halves, Richardson and perturbation weights account for their actual lengths.
Accept against atol+rtol*max(norm(old),norm(new)).
These are local estimates, not a guarantee on the final/global error or errors
in directions invisible to the current reduced bases.

Full-step truncation budget is truncation_fraction*(atol+rtol*norm(old)); each
half gets half that budget. All three solves must meet their budgets. Typed
local failures, excessive truncation, and excessive estimated errors trigger
rejection without changing the accepted state. `maxsteps` counts all attempts;
`maxreject` counts consecutive rejections. Limits raise AdaptiveIntegrationError
with the last accepted state and attempt history. A cap that cannot represent
the solution may remain inadequate at arbitrarily small steps.

`saveat` is an increasing vector/range/tuple of times within tspan; endpoints
are included. Steps end at saved times; no dense interpolation is used, and
changing saveat can change the internal mesh. A final step to a saved time may
be shorter than dtmin, but cannot be retried below dtmin. Half steps must also
be representable at the current absolute time.

Return `times`, observed `values`, node-by-sample `ranks`, `diagnostics` for
every accepted/rejected attempt, `stats`, and `final_state`. `observe` receives
an independent TTN snapshot, protecting the live state from callback mutation.
By default values stores TTNs; supply an observable to limit history memory.
Inputs and saved snapshots are independent. Options are validated even when
tspan has zero length. Krylov tolerance has the same meaning as in the step API.

Set `record_timing=true` for wall-clock seconds in `timings` and each attempt's
`diagnostics[k].timings`. Totals separate `coarse_bug` (error-estimation step),
`accepted_update` / `rejected_update` (two half steps), `difference` (TTN
difference construction plus QR/norm), `candidate_norm`, `observations`
(snapshot copying, callback and saved ranks), and `other` (remaining driver
work). `total` covers the driver, including setup and result aggregation.
Per-attempt `fine_bug` includes both half steps actually attempted. Failed
solves are timed too. Timings include GC and compilation if these occur;
warm up before performance comparisons. With timing disabled these fields
are `nothing` and no clocks are read.

The full step and first half step share old-state orthogonalization, operator
contractions and complement QR data. This preparation is charged once to
`coarse_bug`; each solve owns its updated bases. The second half step prepares
its changed input independently. No preparation survives the trial.
"""
function adaptive_evolve(initial::TTNTensor, operator::SumOfProducts, tspan;
                         dt, saveat=nothing, atol=1e-6, rtol=1e-4,
                         truncation_fraction=0.1, dtmin=0.0, dtmax=Inf,
                         maxsteps=100000, maxreject=20, observe=copy,
                         maxrank=typemax(Int), conserved=nothing,
                         krylov_tol=1e-12, krylovdim=30, krylov_maxiter=100,
                         correct_invariant::Bool=true, record_timing::Bool=false,
                         method=:augmented, threaded::Bool=true,truncation=:weighted)
    method === :augmented || method === :parallel ||
        throw(ArgumentError("method must be :augmented or :parallel"))
    _check_truncation_weighting(truncation)
    started = record_timing ? time_ns() : UInt64(0)
    observations_timer = record_timing ? _AdaptiveTimings() : nothing
    grid = _save_grid(tspan, saveat)
    abs_tol, rel_tol = _nonnegative_float(atol, "atol"), _nonnegative_float(rtol, "rtol")
    abs_tol > 0 || rel_tol > 0 || throw(ArgumentError("at least one error tolerance must be positive"))
    fraction = _nonnegative_float(truncation_fraction, "truncation_fraction")
    0 < fraction < 1 || throw(ArgumentError("truncation_fraction must lie strictly between zero and one"))
    proposal = _nonnegative_float(dt, "dt")
    proposal > 0 || throw(ArgumentError("dt must be positive in Float64"))
    minimum_step = _nonnegative_float(dtmin, "dtmin")
    dtmax isa Real && !isnan(dtmax) && dtmax > 0 || throw(ArgumentError("dtmax must be positive"))
    maximum_step = Float64(dtmax)
    maximum_step > 0 && minimum_step <= maximum_step && proposal >= minimum_step ||
        throw(ArgumentError("require 0 <= dtmin <= dtmax and dt >= dtmin"))
    for (value, name) in ((maxsteps, "maxsteps"), (maxreject, "maxreject"))
        value isa Integer && 1 <= value <= typemax(Int) || throw(ArgumentError("$name must be a positive Int"))
    end
    _check_bug_options(0.0, maxrank, krylov_tol, krylovdim, krylov_maxiter)
    _check_operator(operator, initial)
    c = _canonical_conserved(initial, conserved)
    if !isnothing(c)
        ranks = tree_ranks(c)
        all(node -> node == c.tree.root || ranks[node] <= maxrank, c.tree.postorder) ||
            throw(ArgumentError("maxrank cannot hold the conserved directions"))
    end
    options = (; maxrank, conserved=c, krylov_tol, krylovdim, krylov_maxiter, correct_invariant,truncation)
    state = copy(initial)
    state_norm = norm(state)
    isfinite(state_norm) || throw(ArgumentError("initial state norm must be finite"))
    sample = _measure_adaptive(observations_timer,Val(:observations)) do
        (;value=observe(copy(state)),ranks=tree_ranks(state))
    end
    values = [sample.value]
    sample_ranks = [sample.ranks]
    history = NamedTuple[]
    accepted, rejected, consecutive_rejections = 0, 0, 0
    time = first(grid)
    proposal = min(proposal, maximum_step)

    for stop in grid[2:end]
        while time < stop
            length(history) < maxsteps || _adaptive_stop(:maxsteps, time, proposal, state, history)
            target = proposal >= stop-time ? stop : time+proposal
            h = target-time
            # Rounding time+proposal can cross a hard dtmax (or fall below
            # dtmin). Choose a neighboring representable endpoint first.
            if h > maximum_step
                target = prevfloat(target)
                h = target-time
            end
            if h < minimum_step && target != stop
                target = nextfloat(target)
                h = target-time
            end
            time < target <= stop && 0 < h <= maximum_step &&
                (target == stop || h >= minimum_step) ||
                _adaptive_stop(:time_resolution, time, h, state, history)
            middle = time+h/2
            time < middle < target || _adaptive_stop(:time_resolution, time, h, state, history)
            split_ratio = (middle-time)/(target-middle)
            richardson_factor = (split_ratio + inv(split_ratio))/2
            tolerance = abs_tol + rel_tol*state_norm
            isfinite(tolerance) || throw(ArgumentError("error tolerance overflowed; rescale tolerances/state"))
            budget = fraction*tolerance
            timer = record_timing ? _AdaptiveTimings() : nothing
            trial = _adaptive_trial(state, operator, middle-time, target-middle, budget, options, timer;
                                    method,threaded)
            reason = trial.reason
            difference = error_estimate = error_ratio = Inf
            truncation_error = correction_error = 0.0
            candidate_norm = state_norm
            if reason == :candidate
                candidate_norm = _measure_adaptive(timer,Val(:candidate_norm)) do
                    norm(trial.candidate)
                end
                tolerance = abs_tol + rel_tol*max(state_norm, candidate_norm)
                difference = trial.difference
                truncation_error = richardson_factor*trial.coarse.truncation.discarded_bound +
                    (richardson_factor+1)*(trial.half1.truncation.discarded_bound + trial.half2.truncation.discarded_bound)
                correction_error = richardson_factor*trial.coarse.correction_norm +
                    (richardson_factor+1)*(trial.half1.correction_norm + trial.half2.correction_norm)
                error_estimate = richardson_factor*difference + truncation_error + correction_error
                error_ratio = tolerance == 0 ? (error_estimate == 0 ? 0.0 : Inf) : error_estimate/tolerance
                reason = isfinite(candidate_norm) && isfinite(tolerance) && isfinite(error_ratio) && error_ratio <= 1 ?
                    :accepted : :error
            end
            keep = reason == :accepted
            push!(history, (; time, t_end=target, dt=h, accepted=keep, reason,
                difference, richardson_factor, error_estimate, error_ratio, tolerance,
                truncation_budget=budget, truncation_error, correction_error,
                stage=trial.stage, local_failure=trial.local_failure,
                coarse=trial.coarse, half1=trial.half1, half2=trial.half2,
                timings=_trial_timings(timer)))
            if keep
                state = trial.candidate
                state_norm = candidate_norm
                time = target
                accepted += 1
                consecutive_rejections = 0
                factor = error_ratio == 0 ? 2.0 : clamp(0.9/sqrt(error_ratio), 0.2, 2.0)
                proposal = clamp(h*factor, minimum_step, maximum_step)
            else
                rejected += 1
                consecutive_rejections += 1
                consecutive_rejections < maxreject || _adaptive_stop(:maxreject, time, h, state, history)
                factor = reason == :error && isfinite(error_ratio) ? clamp(0.9/sqrt(error_ratio), 0.2, 0.8) : 0.5
                proposal = h*factor
                proposal >= minimum_step || _adaptive_stop(:dtmin, time, h, state, history)
            end
        end
        sample = _measure_adaptive(observations_timer,Val(:observations)) do
            (;value=observe(copy(state)),ranks=tree_ranks(state))
        end
        push!(values,sample.value)
        push!(sample_ranks,sample.ranks)
    end
    ranks = hcat(sample_ranks...)
    timings = _evolution_timings(history,observations_timer,started)
    return (; times=grid, values, ranks, diagnostics=history,
            stats=(; accepted, rejected, attempts=length(history)), final_state=state,timings)
end
