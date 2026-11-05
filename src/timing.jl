# Optional wall-clock accounting. No clock reads on the default path.
mutable struct _AdaptiveTimings
    coarse_bug::Float64
    fine_bug::Float64
    difference::Float64
    candidate_norm::Float64
    observations::Float64
end

_AdaptiveTimings() = _AdaptiveTimings(0.,0.,0.,0.,0.)

@inline _measure_adaptive(f, ::Nothing, ::Val) = f()

function _measure_adaptive(f, timer::_AdaptiveTimings, ::Val{field}) where {field}
    started = time_ns()
    try
        return f()
    finally
        # Include time spent in failed local solves before rejection/retry.
        elapsed = Float64(time_ns()-started)*1e-9
        setfield!(timer,field,getfield(timer,field)+elapsed)
    end
end

_trial_timings(::Nothing) = nothing
_trial_timings(t::_AdaptiveTimings) =
    (; t.coarse_bug,t.fine_bug,t.difference,t.candidate_norm)

_evolution_timings(history, ::Nothing, started) = nothing

function _evolution_timings(history, observations::_AdaptiveTimings, started)
    coarse_bug = accepted_update = rejected_update = difference = candidate_norm = 0.0
    for d in history
        t = d.timings
        coarse_bug += t.coarse_bug
        if d.accepted
            accepted_update += t.fine_bug
        else
            rejected_update += t.fine_bug
        end
        difference += t.difference
        candidate_norm += t.candidate_norm
    end
    observed = observations.observations
    total = Float64(time_ns()-started)*1e-9
    other = max(0.,total-coarse_bug-accepted_update-rejected_update-difference-candidate_norm-observed)
    return (;coarse_bug,accepted_update,rejected_update,difference,candidate_norm,
            observations=observed,other,total)
end
