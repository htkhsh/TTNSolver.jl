# Experimental tree operators

`TreeTensorOperator(A, tree)` compiles a `SumOfProducts` into a TTNO on the
same ordered tree as the state. Equal local matrices and equal subtree products
share channels. The default compilation is exact apart from roundoff; it does
not perform numerical truncation. Optional `atol` and `maxrank` use weighted
operator-TTN compression and expose the discarded Frobenius bound as
`W.discarded_bound`. A rank cap can make this bound exceed `atol`.

```julia
W = TreeTensorOperator(A, x.tree)
y = apply(W, x)
@show tree_ranks(W)
endpoint, info = tamen_step(x, A, 0.1; residual_backend=:ttno)
```

`apply` contracts local operator tensors with the state and QR-factorizes each
result subtree once. It returns independently owned, root-orthogonal state
storage. It does not truncate that state. `dense(W)` is for small tests only.
The physical operator index is `(output,input)`, with output fastest; physical
mode ordering continues to follow `tree.leaf_order`.

The tAMEn option changes **global residual certification only**. The default is
`residual_backend=:sop`; the local/Etter ALS, auxiliary residual enrichment,
trace constraints, time discretization and acceptance tolerance stay in place.
The TTNO is built lazily at the first global check and reused within that
interval, never across changed operators. After structural compilation it uses
an absolute Frobenius budget `target / (2000 * max(norm(x_first_check), 1))` to
compress the operator. The actual discarded bound is multiplied by the current
iterate norm at every check, including later larger iterates. Thus growth can
make a certificate fail, but cannot silently remove the operator-error allowance.
`info.residual_operator_ranks` and `info.residual_operator_discarded_bound`
report the compiled operator. Compilation time is included in
`timings.residual_global_seconds`.

The alternate certificate forms `r = b-W*x` explicitly and compresses it once.
If W was numerically compressed, its bound is

```text
norm(compressed r) + residual_discarded_bound + W.discarded_bound * norm(x).
```

This uses the Frobenius-to-operator-norm inequality and the triangle inequality.
As in the SoP certificate, floating-point roundoff is excluded. The small
residual is not obtained by subtracting three large squared norms.

This first implementation merges equal subtrees; it does **not** implement
PyTreeNet's symbolic Gaussian elimination/minimum-vertex-cover construction.
Operator and state ranks multiply before QR, so an unfavorable tree or a large
state rank can make one-pass application substantially more expensive than
streamed SoP certification. Before allocation, `apply` rejects a predicted
local QR input larger than 1 GiB; standalone callers may explicitly change
`max_intermediate_bytes`. This limits one input array, not total peak memory.
For tAMEn, use `:sop` if this guard rejects the TTNO path. There is no automatic
backend selection. TTNO
application is serial; `threaded=true` currently parallelizes the subsequent
canonical residual sum, not the TTNO contraction itself. Local BUG integration
still uses its specialized matrix-free SoP contractions.

The consulted implementation is
[PyTreeNet TTNO construction](../ref/PyTreeNet-main/pytreenet/ttno/ttno_class.py)
and its [state diagrams](../ref/PyTreeNet-main/pytreenet/ttno/state_diagram.py).
The Julia implementation uses the project's existing TTN algebra and QR/SVD
routines; third-party source was not copied. Benchmark conditions and results
are recorded in [TTNDynamics](../../TTNDynamics/docs/reference_optimization.md).
