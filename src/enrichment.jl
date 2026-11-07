"""
    enrich_basis(state, direction)

Return an independent TTN for the same state, with every non-root subtree
basis enlarged to contain the direction's subtree basis. This is a basis
union, not addition of direction to the wavefunction. Original state bases
are retained first; amplitudes and phase are preserved to roundoff. Inputs
must have identical ordered trees and physical dimensions and are unchanged.

No global tensor is expanded. Bond ranks are at most the sums of input ranks
(bounded by the local space). Added directions initially have zero weight;
truncating before propagation can discard them. Useful for bootstrapping BUG
when a product state's reduced equations cannot discover coupling directions.
It is not an error estimator and supplies no convergence guarantee.
"""
function enrich_basis(state::TTNTensor, direction::TTNTensor; threaded=false)
    _check_layout(state,direction)
    tree = state.tree
    _isleaf(tree,tree.root) && return copy(state)
    x,y = orthogonalize!(copy(state); threaded),orthogonalize!(copy(direction); threaded)
    result = copy(x)
    leaves,cores = _node_tensors(result)
    overlaps_x = Vector{Matrix{ComplexF64}}(undef,length(tree.children))
    overlaps_y = similar(overlaps_x)
    # Each node needs only its children's overlaps; sibling subtrees may run
    # concurrently with the same arithmetic.
    _each_subtree(tree; threaded) do node
        if _isleaf(tree,node)
            X,Y = x.leaves[node],y.leaves[node]
            Q = _orth_columns(X,Y)
            leaves[node] = Q
        else
            left,right = tree.children[node]
            projected_x = _project_children(x.cores[node],overlaps_x[left],overlaps_x[right])
            if node == tree.root
                cores[node] = projected_x
                return
            end
            projected_y = _project_children(y.cores[node],overlaps_y[left],overlaps_y[right])
            X,Y = _basis_matrix(projected_x),_basis_matrix(projected_y)
            Q = _orth_columns(X,Y)
            cores[node] = reshape(Matrix(transpose(Q)),size(Q,2),size(projected_x,2),size(projected_x,3))
        end
        overlaps_x[node] = Q'*X
        overlaps_y[node] = Q'*Y
    end
    _store_node_tensors!(result, leaves, cores)
    return result
end
