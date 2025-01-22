function get_ab(dc::Int, D::Int, a::Vector, b::Vector, ind::Int, tree::Any) #get the parameters (a,b) of the tree nodes
    @inbounds if !(tree isa Leaf)
        a[ind] = tree.featid
        b[ind] = tree.featval
        if dc <= D
            node_l = 2 * ind
            node_r = 2 * ind + 1
            get_ab(dc + 1, D, a, b, node_l, tree.left)
            get_ab(dc + 1, D, a, b, node_r, tree.right)
        end
    end
    return a, b
end
