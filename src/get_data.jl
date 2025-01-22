function get_data(X::AbstractMatrix{Float64}, Y::AbstractVector{Int}, i::Int, A::Vector, B::Vector)   #get the data partitioned to each node
    n, p = size(X)
    ancesters = [floor(Int32, i / 2^j) for j in floor(Int32, log2(i)):-1:0] #the routing path
    size_ancs = size(ancesters, 1)
    selected = trues(n)
    for index in 1:n
        for i in 1:(size_ancs-1) # 1,2
            if ancesters[i+1] == 2 * ancesters[i] # left_node, ax<b
                if A[ancesters[i]] == 0
                    selected[index] = false
                    continue
                end
                if selected[index] #  X[index, :][a[:, ancesters[i]]][1] >= b[ancesters[i]]
                    if X[index, A[ancesters[i]]] >= B[ancesters[i]]
                        selected[index] = false
                    end     # end if
                end         # end if 
            else            # right_node, ax>=b
                if A[ancesters[i]] == 0
                    continue
                end
                if selected[index] #  X[index, :][a[:, ancesters[i]]][1] < b[ancesters[i]]
                    if X[index, A[ancesters[i]]] < B[ancesters[i]]
                        selected[index] = false
                    end      # end if
                end          # end if 
            end              # end if         
        end                  # end for
    end                      # end if
    return copy(X[selected, :]), copy(Y[selected])
end