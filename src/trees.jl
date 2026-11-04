"""
    BinaryTree(children; root=length(children), leaf_order=nothing)

An ordered rooted binary tree. `children[node] == (0, 0)` marks a leaf;
otherwise it contains distinct left/right child IDs. Every node must be
reachable from `root` and have exactly one parent, except the root.

`leaf_order` specifies physical tensor axis order and defaults to ascending
leaf IDs. Input arrays are copied. Treat the stored topology and traversal
arrays as read-only; construct a new tree to change topology.
"""
struct BinaryTree
    children::Vector{NTuple{2,Int}}
    parent::Vector{Int}
    root::Int
    leaf_order::Vector{Int}
    postorder::Vector{Int}

    function BinaryTree(children::AbstractVector; root::Integer=length(children),
                        leaf_order=nothing)
        n = length(children)
        n > 0 || throw(ArgumentError("a tree needs at least one node"))
        1 <= root <= n || throw(ArgumentError("root must be a valid node ID"))
        all(pair -> pair isa Tuple && length(pair) == 2 &&
                    all(id -> id isa Integer, pair), children) ||
            throw(ArgumentError("children must be pairs of integer node IDs"))
        pairs = NTuple{2,Int}[(Int(pair[1]), Int(pair[2])) for pair in children]
        parent = zeros(Int, n)
        for (node, (left, right)) in enumerate(pairs)
            (left, right) == (0, 0) && continue
            1 <= left <= n && 1 <= right <= n && left != right ||
                throw(ArgumentError("node $node needs two distinct valid children"))
            for child in (left, right)
                parent[child] == 0 ||
                    throw(ArgumentError("node $child has multiple parents"))
                parent[child] = node
            end
        end
        parent[root] == 0 || throw(ArgumentError("root must not have a parent"))

        visited = falses(n)
        postorder = Int[]
        function visit(node)
            visited[node] && throw(ArgumentError("tree contains a cycle"))
            visited[node] = true
            if pairs[node] != (0, 0)
                visit(pairs[node][1])
                visit(pairs[node][2])
            end
            push!(postorder, node)
        end
        visit(Int(root))
        all(visited) || throw(ArgumentError("all nodes must be connected to the root"))

        leaves = findall(==((0, 0)), pairs)
        order = isnothing(leaf_order) ? leaves : collect(leaf_order)
        all(id -> id isa Integer, order) && sort(order) == leaves ||
            throw(ArgumentError("leaf_order must contain every leaf ID exactly once"))
        return new(pairs, parent, Int(root), Int.(order), postorder)
    end
end

"""
    balanced_tree(nleaves)

Build a balanced binary tree with leaves numbered `1:nleaves` in physical
order. A single leaf is a valid tree and is also its root.
"""
function balanced_tree(nleaves::Integer)
    nleaves >= 1 || throw(ArgumentError("nleaves must be positive"))
    children = fill((0, 0), nleaves)
    function build(first, last)
        first == last && return first
        middle = first + (last - first) ÷ 2
        left = build(first, middle)
        right = build(middle + 1, last)
        push!(children, (left, right))
        return length(children)
    end
    root = build(1, Int(nleaves))
    return BinaryTree(children; root)
end
