function gen_splits(X::AbstractMatrix{Float64}, Y::AbstractVector{Int}) #generate the splits of the features
    n, p = size(X)
    splits = Vector{Vector{Float64}}(undef, p)
    group = Vector{Vector{Int}}(undef, p)
    for i in 1:p
        unique_vals = sort(unique(X[:, i]))
        valind = sortperm(X[:, i])
        vals = X[valind, i]
        Yvals = Y[valind]
        n_unique = length(unique_vals)
        groupnumb = zeros(Int, n_unique)
        if n_unique <= 1
            splits[i] = [unique_vals[1]]
            group[i] = [n]
        else
            splits_i = (unique_vals[1:end-1] + unique_vals[2:end]) / 2
            splits_i = vcat(unique_vals[1] - 1.0, splits_i, unique_vals[end] + 1.0)

            for t in 1:n_unique
                groupnumb[t] = searchsortedfirst(vals, unique_vals[t])
            end
            splits[i] = splits_i
            group[i] = groupnumb
        end
    end
    return splits, group
end


