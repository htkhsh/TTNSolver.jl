# Pre-optimization termwise assembly; test/benchmark oracle only.
function reference_tamen_residual(A,x,b,budget,maxrank)
    residual = copy(b)
    bound = 0.0
    perterm = budget / max(1,length(A.terms))
    for term in A.terms
        iszero(term.coefficient) && continue
        acted = copy(x)
        for (mode,factor) in term.factors
            node = x.tree.leaf_order[mode]
            acted.leaves[node] = factor * acted.leaves[node]
        end
        acted.cores[x.tree.root] .*= term.coefficient
        residual = TTNSolver._tamen_add(residual,acted,-1)
        cut = truncate!(residual;atol=perterm,maxrank)
        bound += cut.discarded_bound
    end
    return residual, norm(residual)+bound, bound
end
