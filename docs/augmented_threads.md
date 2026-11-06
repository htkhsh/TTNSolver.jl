# Augmented BUG threading

`augmented_bug_step(...; threaded=true)` updates independent sibling subtrees
on Julia worker threads. Each parent Galerkin equation uses the updated child
bases, so it starts after both children finish. This preserves the recursive
augmented BUG algorithm [1]. It differs from parallel BUG [2], whose local
equations all use old bases and can run concurrently across all tree levels.

```julia
y, info = augmented_bug_step(x, L, h; threaded=true)
trajectory = adaptive_evolve(x, L, (0., 1.); dt=0.01, threaded=true)
```

Start Julia with `julia --threads=4 ...`. Both APIs default to `threaded=true`;
`threaded=false` executes the same formula sequentially. One-worker execution
and single-leaf trees fall back to serial execution. `info.method` is
`:augmented`; `info.threaded` reports use of the concurrent subtree path.
The adaptive diagnostics retain these fields in `coarse`, `half1`, `half2`.

Each branch owns its local Krylov workspace and its node's vector slots.
No worker mutates shared TTN dictionaries. Results are published sequentially
after the tree update; local statistics are reduced in a deterministic order.
All child tasks, including descendants, are joined before returning a failure.
The original exception type is preserved, so adaptive stepping can reject
`LocalSolveFailure`; programming errors take priority over local solve failures.
Inputs and reusable preparations remain read-only.

Initial QR/environment preparation and final protected rank truncation remain
serial. Root solves, uneven subtree work, task overhead and allocation/GC can
limit speedup. Set BLAS threads explicitly when measuring Julia concurrency;
the model scaling benchmark uses one BLAS and one GC worker to hold them fixed.

The regression suite checks complex states, arbitrary node numbering, nested
subtrees, full leaves, single-leaf and zero states, conservation without root
correction, concurrent preparation reuse, joined failures, adaptive rejection,
and fixed/adaptive HEOM and TFD forwarding. For measured end-to-end scaling,
see the TTNDynamics `benchmarks/augmented_threads/` runner and report.

## References

1. G. Ceruti, C. Lubich and D. Sulz, *Rank-adaptive time integration of tree
   tensor networks*, SIAM J. Numer. Anal. 61(1), 194–222 (2023),
   [doi:10.1137/22M1473790](https://doi.org/10.1137/22M1473790).
2. G. Ceruti, J. Kusch, C. Lubich and D. Sulz, *A Parallel Basis Update and
   Galerkin Integrator for Tree Tensor Networks*, SIAM J. Sci. Comput. 48(1),
   A27–A48 (2026), [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502).
