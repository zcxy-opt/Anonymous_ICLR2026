function mh_cart(X::AbstractMatrix{Float64}, Y::AbstractVector{Int}, tree_depth::Int, epsilon=0.0::Float64, alpha=0.0::Float64, batch=1.0::Float64) 
    branch_size, leaf_size = 2^tree_depth - 1, 2^tree_depth
    min_cost = sum(Y .!= mode(Y))
    start_time = time()
    end_time = time() - start_time
    println(" MH_time: ", end_time, " Optimal value: ", 1 - min_cost / length(Y), " Depth: ", 0)
    best_A, best_B = zeros(Int, branch_size), zeros(Float64, branch_size)
    best_C = zeros(Float64, leaf_size)
    mhind = Array(1:max(2^(tree_depth - 1) - 1, 1))
    #mhind = [1]
    for node in mhind   #iterate over the nodes in the moving horizon process
        A, B = copy(best_A), copy(best_B)
        subtree_depth = floor(Int32, log2(node)) + 1
        sub_tree_size = 2^(tree_depth - subtree_depth + 1) - 1
        sub_a, sub_b = zeros(Int, sub_tree_size), zeros(Float64, sub_tree_size)
        sub_c = zeros(Int, 2^(tree_depth - subtree_depth + 1))
        X_i, Y_i = get_data(X, Y, node, A, B)
        @inbounds if size(X_i, 1) > 1 && length(unique(Y_i)) > 1 #substitute the node parameters corresponding to the subtree
            subnode = []
            sub_a, sub_b, sub_c, sub_sign = each_node(X_i, Y_i, node, tree_depth, epsilon, alpha, batch)  #evaluate the subtree
            @simd for d = 0:(tree_depth-subtree_depth)      #update the tree parameters
                append!(subnode, vec(2^(d)*node:2^(d)*node+2^(d)-1))
                idx_range = (2^d*node):(2^d*node+2^d-1)
                A[idx_range] = copy(sub_a[2^d:2^(d+1)-1])
                B[idx_range] = copy(sub_b[2^d:2^(d+1)-1])
            end

            if sub_sign  #if the subtree is the optimal, remove the nodes from the mhind
                mhind = filter!(x -> x != Array(subnode), mhind)
            end
        end
        C, cost = route(X, Y, A, B)
        if cost === 0 # if the cost is 0, return the global optimal result
            return A, B, C, 0
        end

        if cost < min_cost  #update the global optimal result
            best_A, best_B, best_C, min_cost = copy(A), copy(B), copy(C), cost
        end
        if floor(log(2, node + 1)) == log(2, node + 1)
            end_time = time() - start_time
            println(" MH_time: ", end_time, " Optimal value: ", 1 - min_cost / length(Y), " Depth: ", floor(log(2, node)) + 1, " node: ", node)
        end
    end
    return best_A, best_B, best_C, min_cost
end





