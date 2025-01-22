function eva(X::AbstractMatrix{Float64}, Y::AbstractVector{Int}, A::Vector, B::Vector, C::Vector) #evaluate the misclassigication loss of the tree
    n, p = size(X)
    branch_size = size(A, 1)
    num_error = 0
    for i in 1:n
        node = 1
        while node <= branch_size
            feature = A[node]
            if feature == 0
                node = 2 * node + 1
                continue
            end
            if X[i, feature] < B[node]
                node = 2 * node
            else
                node = 2 * node + 1
            end
        end

        if Y[i] != C[node - branch_size]
            num_error += 1
        end
    end
    return num_error
end




