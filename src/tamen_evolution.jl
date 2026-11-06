"""Adaptive tAMEn stopped at a limit. `state` and `time` are the last accepted
state/time; `diagnostics` includes rejected attempts. No failed iterate is adopted.
"""
struct TAMEnIntegrationError <: Exception
    reason::Symbol
    time::Float64
    dt::Float64
    state::TTNTensor
    diagnostics::Vector{NamedTuple}
end
Base.showerror(io::IO,e::TAMEnIntegrationError) =
    print(io,"adaptive tAMEn stopped: ",e.reason," at t=",e.time,", dt=",e.dt,
          "; inspect exception.state and exception.diagnostics")

function _tamen_stop(reason,time,dt,state,history)
    throw(TAMEnIntegrationError(reason,time,dt,copy(state),history))
end

function _compact_tamen(info,store)
    record = Base.structdiff(info,(;warm_start=info.warm_start))
    return store ? record : Base.structdiff(record,(;solution=record.solution))
end

"""
    tamen_evolve(initial, generator, tspan; dt, saveat=nothing,
        atol=1e-8, rtol=1e-6, time_points=8, time_scheme=:cheb,
        time_error_damp=100, solve_rtol=1e-10, solve_atol=0,
        dtmin=0, dtmax=Inf, maxsteps=100000, maxreject=20,
        observe=copy, store_solutions=false, warm_start=true, workspace=nothing, kwargs...)

Adaptive intervals for an autonomous homogeneous linear ODE. A refined-grid
integrated defect in the final spatial basis estimates temporal error; the
initial projection error is included. This follows TT tAMEn's reduced-space
time-control approach, not BUG step doubling. It is a local estimator, not a
global error guarantee. Accept only converged linear solves with
`time_error_damp*error <= atol + rtol*scale`. `time_error_damp=0` disables temporal
rejection (linear convergence is still required). Interval growth is capped at
2 and uses a 0.5 safety factor, with power 1/nt for Chebyshev and 1/2 for CN.

`solve_rtol/solve_atol` control the independent space-time linear residual.
Other solver keywords are passed to `tamen_step`. Rejection retries from the
last accepted state. Limits throw `TAMEnIntegrationError`. `maxsteps` counts
all attempts; `maxreject` counts consecutive rejections.

`saveat` is independent of the internal interval grid: outputs use temporal
interpolation, and include both endpoints of tspan. The observer receives an
independent TTN. Return `times`, observed `values`, sample `ranks`, `final_state`,
accepted interval boundaries, attempt diagnostics and accepted/rejected counts.
Full space-time solutions are retained only with `store_solutions=true`.
One workspace lease covers the entire run, including observers.
With default `warm_start=true`, the last accepted space-time solution and
auxiliary residual initialize the next interval. Rejected trials do not replace
this hint. Set `warm_start=false` for cold interval initialization. Hints are
local to one evolution call, never stored in the workspace or history (even
when `store_solutions=true`), so retained hint storage is independent of the
number of intervals. `store_warm_start` is controlled by this driver.
"""
function tamen_evolve(initial::TTNTensor,generator::SumOfProducts,tspan;
                      workspace=nothing,kwargs...)
    return _with_tamen_workspace(workspace) do work
        _tamen_evolve(initial,generator,tspan;workspace=work,kwargs...)
    end
end

function _tamen_evolve(initial,generator,tspan;workspace,dt,saveat=nothing,
                       atol=1e-8,rtol=1e-6,time_points=8,time_scheme=:cheb,
                       time_error_damp=100.,solve_rtol=1e-10,solve_atol=0.,
                       dtmin=0.,dtmax=Inf,maxsteps=100000,maxreject=20,
                       observe=copy,store_solutions::Bool=false,warm_start::Bool=true,kwargs...)
    for key in (:strict,:estimate_time_error,:store_warm_start)
        haskey(kwargs,key) && throw(ArgumentError("$key is controlled by tamen_evolve"))
    end
    _check_operator(generator,initial)
    times = _save_grid(tspan,saveat)
    abs_tol,rel_tol = _nonnegative_float(atol,"atol"),_nonnegative_float(rtol,"rtol")
    abs_tol>0 || rel_tol>0 || throw(ArgumentError("one temporal tolerance must be positive"))
    damp = _nonnegative_float(time_error_damp,"time_error_damp")
    proposal = _nonnegative_float(dt,"dt")
    proposal>0 || throw(ArgumentError("dt must be positive"))
    minimum_step = _nonnegative_float(dtmin,"dtmin")
    dtmax isa Real && !isnan(dtmax) && dtmax>0 || throw(ArgumentError("dtmax must be positive"))
    maximum_step = Float64(dtmax)
    maximum_step>0 && minimum_step<=maximum_step && proposal>=minimum_step ||
        throw(ArgumentError("require 0 <= dtmin <= dtmax and dt >= dtmin"))
    for (key,value) in ((:maxsteps,maxsteps),(:maxreject,maxreject))
        value isa Integer && 1<=value<=typemax(Int) || throw(ArgumentError("$key must be a positive Int"))
    end
    scheme = _time_scheme(time_scheme)
    _tamen_grid(time_points,1.,scheme) # validate even for an empty tspan
    order = scheme==:cheb ? time_points : 2
    state = copy(initial)
    ranks_saved = [tree_ranks(state)]
    values = [observe(copy(state))]
    time,finish = first(times),last(times)
    accepted_times = [time]
    history = NamedTuple[]
    accepted = rejected = consecutive = 0
    accepted_hint = nothing
    sample = 2
    proposal = min(proposal,maximum_step)
    while time < finish
        length(history)<maxsteps || _tamen_stop(:maxsteps,time,proposal,state,history)
        target = proposal>=finish-time ? finish : time+proposal
        h = target-time
        if h>maximum_step
            target = prevfloat(target)
            h = target-time
        end
        if h<minimum_step && target<finish
            target = nextfloat(target)
            h = target-time
        end
        time<target<=finish && h<=maximum_step && (target==finish || h>=minimum_step) ||
            _tamen_stop(:time_resolution,time,h,state,history)
        info = nothing
        candidate = nothing
        reason = :linear_residual
        started = time_ns()
        try
            candidate,info = _tamen_step(state,generator,h;
                workspace,time_scheme=scheme,time_points,rtol=solve_rtol,atol=solve_atol,
                strict=false,estimate_time_error=true,warm_start=accepted_hint,
                store_warm_start=warm_start,kwargs...)
        catch error
            error isa TAMEnConvergenceError || rethrow()
            reason = :local_gmres
            info = (;converged=false,failure=error.reason,local_diagnostics=error.diagnostics)
        end
        error_ratio = tolerance = error_estimate = Inf
        if info.converged
            tolerance = abs_tol+rel_tol*info.time_error.scale
            isfinite(tolerance) || throw(ArgumentError("temporal tolerance overflow"))
            error_estimate = iszero(damp) ? 0.0 : damp*info.time_error.absolute
            error_ratio = iszero(tolerance) ? (iszero(error_estimate) ? 0.0 : Inf) : error_estimate/tolerance
            reason = isfinite(error_ratio) && error_ratio<=1 ? :accepted : :time_error
        end
        keep = reason==:accepted
        solve = hasproperty(info,:solution) ? _compact_tamen(info,store_solutions) : info
        push!(history,(;time,t_end=target,dt=h,accepted=keep,reason,error_estimate,
            error_ratio,tolerance,solve,seconds=(time_ns()-started)/1e9))
        if keep
            # Commit hint only after both linear and temporal acceptance. It
            # owns arrays distinct from the diagnostic solution and outputs.
            accepted_hint = warm_start ? info.warm_start : nothing
            # Observations cannot mutate the accepted iterate or change dt.
            while sample<=length(times) && times[sample]<=target
                snapshot = times[sample]==target ? copy(candidate) :
                    time_snapshot(info.solution,clamp(times[sample]-time,0.,h))
                push!(ranks_saved,tree_ranks(snapshot))
                push!(values,observe(snapshot))
                sample += 1
            end
            state = candidate
            time = target
            push!(accepted_times,time)
            accepted += 1
            consecutive = 0
            factor = iszero(error_ratio) ? 2.0 : clamp(0.5*error_ratio^(-1/order),0.1,2.0)
            proposal = clamp(h*factor,minimum_step,maximum_step)
        else
            rejected += 1
            consecutive += 1
            consecutive<maxreject || _tamen_stop(:maxreject,time,h,state,history)
            factor = reason==:time_error && isfinite(error_ratio) ?
                clamp(0.5*error_ratio^(-1/order),0.1,0.8) : 0.5
            proposal = h*factor
            proposal>=minimum_step || _tamen_stop(:dtmin,time,h,state,history)
        end
    end
    return (;times,values,ranks=hcat(ranks_saved...),final_state=state,
            accepted_times,diagnostics=history,accepted,rejected)
end
