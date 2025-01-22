function get_c(X::AbstractMatrix{Float64}, Y::AbstractVector{Int}, A::Vector, B::Vector)  #get the class labels of the leaf nodes
    n, p = size(X)
    branch_size = size(A, 1)
    leaf_size = size(A, 1) + 1
    Z = zeros(Int, n)
    for i in 1:n
        node = 1
        while node <= branch_size
            if A[node] == 0
                node = 2 * node + 1
                continue
            end
            if X[i, A[node]] < B[node]
                node = 2 * node
            else
                node = 2 * node + 1
            end
        end
        Z[i] = node - branch_size
    end
    C = zeros(Int, leaf_size)
    for i in 1:leaf_size
        tmp_Z = Z .== i
        if sum(tmp_Z) == 0
            continue
        end
        tmp_Y = Y[tmp_Z]
        C[i] = mode(tmp_Y)
    end

    return C
end