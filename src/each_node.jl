function each_node(X_i::AbstractMatrix{Float64}, Y_i::AbstractArray{Int}, node_i::Int, depth::Int, epsilon=0.0::Float64, alpha=0.0::Float64, batch=1.0::Float64)  # function of the Branch and Reduce method
    signopt = false
    n, p = size(X_i)
    depth_i = floor(Int, log2(node_i)) + 1
    D = depth - depth_i
    min_cost = n
    best_A, best_B = zeros(Int, 2^(D + 1) - 1), zeros(Float64, 2^(D + 1) - 1)
    best_C = zeros(Int, 2^(D + 1))
    splits, group = gen_splits(X_i, Y_i)    # generate splits and group
    for i in 1:p
        splits_i, group_i = splits[i], group[i]
        n_splits = length(splits_i)
        epsilon = epsilon * length(Y_i)
        min_cost, best_A, best_B, best_C = bestsplit(min_cost, best_A, best_B, best_C, X_i, Y_i, splits_i, group_i, Array(1:n_splits), D, i, epsilon, alpha, batch) # function BiEvl to find the best split
    end
    if min_cost === 0           # if the cost is 0, the current subtree is the optimal
        signopt = true
    end
    return best_A, best_B, best_C, signopt
end


@inline function bestsplit(
    min_cost::Int,
    best_A::Vector{Int},
    best_B::Vector{Float64},
    best_C::Vector{Int},
    X_i::AbstractMatrix{Float64},
    Y_i::AbstractVector{Int},
    splits_i::Vector{Float64},
    group_i::Vector{Int},
    indset::AbstractVector{Int},
    D::Int,
    ai::Int,
    epsilon::Float64,
    alpha::Float64,
    batch::Float64
)
    cur_cost = min_cost                                                                 # min_cost is the upper bound as well as the historical minimum cost
    if !isempty(indset) && length(indset) > epsilon
        k = indset[ceil(Int, length(indset) / 2)]                                       #bisection
        split = splits_i[k]
        X_left = @view X_i[X_i[:, ai].<split, :]                                        # split the data
        Y_left = @view Y_i[X_i[:, ai].<split]
        X_right = @view X_i[X_i[:, ai].>=split, :]
        Y_right = @view Y_i[X_i[:, ai].>=split]
        indset = indset[indset.!=k]
        if (batch < 1.0) & (batch > 0.0)                                                # Randomly pick a batch of the data to evaluate the subtree costs
            a1, a2 = shuffle(1:ceil(Int, length(Y_left))), shuffle(1:ceil(Int, length(Y_right)))
            ind1, ind2 = a1[1:ceil(Int, length(a1) * batch)], a2[1:ceil(Int, length(a2) * batch)]
            X_left, Y_left = X_left[ind1, :], Y_left[ind1]
            X_right, Y_right = X_right[ind2, :], Y_right[ind2]
        elseif batch > 1.0 || batch <= 0.0
            error("batch is not feasible")
        end
        tree_left, cost_left, left_a, left_b = evaluate(Y_left, X_left, D, alpha)       # evaluate the left subtree
        tree_right, cost_right, right_a, right_b = evaluate(Y_right, X_right, D, alpha) # evaluate the right subtree
        cur_cost = cost_left + cost_right                                               # calculate the cost of the current subtree
        if cur_cost < min_cost
            best_A[1], best_B[1] = ai, split
            update_best_split!(best_A, best_B, left_a, left_b, right_a, right_b, D)     # update the tree parameters
            min_cost = cur_cost                                                         # update the minimum cost
        end
        delta = max((cur_cost - min_cost), 1)                                            # calculate delta
        left_set = intersect(indset, findall(x -> (x <= group_i[k] - delta), group_i))   # branch the reduced left set
        right_set = intersect(indset, findall(x -> (x >= group_i[k] + delta), group_i))  # branch the reduced right set
        min_cost, best_A, best_B, best_C = bestsplit(min_cost, best_A, best_B, best_C, X_i, Y_i, splits_i, group_i, left_set, D, ai, epsilon, alpha, batch)                            # recursively evaluate the left set
        min_cost, best_A, best_B, best_C = bestsplit(min_cost, best_A, best_B, best_C, X_i, Y_i, splits_i, group_i, right_set, D, ai, epsilon, alpha, batch)                           # recursively evaluate the right set
    end
    return min_cost, best_A, best_B, best_C
end


@inline function evaluate(
    Y::AbstractVector{Int},
    X::AbstractMatrix{Float64},
    D::Int,
    alpha::Float64
)
    a, b = zeros(Int, 2^D - 1), zeros(Float64, 2^D - 1)
    c = zeros(Int, 2^D)
    if length(Y) >= 1
        if D == 0
            cost = sum(Y .!= mode(Y))
            return nothing, cost, [], []
        else
            tree = DecisionTree_modified.build_tree(Y, X, 0, D, 1, 2, alpha)   # build the tree with the data
            if tree.node isa Leaf
                cost = sum(Y .!= tree.node.majority)
                return tree, cost, a, b
            else
                a, b = get_ab(1, D, a, b, 1, tree.node)
                cost = sum(DecisionTree_modified.apply_tree(tree, X) .!= Y)
                return tree, cost, a, b
            end
        end
    else
        return nothing, 0, a, b
    end
end

@inline function update_best_split!(A::Vector, B::Vector, left_a::Vector, left_b::Vector, right_a::Vector, right_b::Vector, depth_diff::Int) # update the tree parameters   
    @simd for d in 1:depth_diff
        A[2^d:3*2^(d-1)-1], A[3*2^(d-1):2^(d+1)-1] = left_a[2^(d-1):2^d-1], right_a[2^(d-1):2^d-1]
        B[2^d:3*2^(d-1)-1], B[3*2^(d-1):2^(d+1)-1] = left_b[2^(d-1):2^d-1], right_b[2^(d-1):2^d-1]
    end
end


