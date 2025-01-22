# Utilities

include("tree.jl")

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels) = Dict(v => k for (k, v) in enumerate(labels))

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::AbstractVector, votes::AbstractVector, weights=1.0)
    label2ind = label_index(labels)
    counts = zeros(Float64, length(label2ind))
    for (i, label) in enumerate(votes)
        if isa(weights, Number)
            counts[label2ind[label]] += weights
        else
            counts[label2ind[label]] += weights[i]
        end
    end
    return counts / sum(counts) # normalize to get probabilities
end

# Applies `row_fun(X_row)::AbstractVector` to each row in X
# and returns a matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::AbstractMatrix)
    N = size(X, 1)
    N_cols = length(row_fun(X[1, :])) # gets the number of columns
    out = Array{Float64}(undef, N, N_cols)
    for i in 1:N
        out[i, :] = row_fun(X[i, :])
    end
    return out
end

function _convert(
    node::treeclassifier.NodeMeta{S}, list::AbstractVector{T}, labels::AbstractVector{T}
) where {S,T}
    if node.is_leaf
        return Leaf{T}(list[node.label], labels[node.region])
    else
        left = _convert(node.l, list, labels)
        right = _convert(node.r, list, labels)
        return Node{S,T}(node.feature, node.threshold, left, right, node.eveloss, node.threbound)
    end
end

function update_using_impurity!(    # update feature importance using impurity
    feature_importance::Vector{Float64}, node::treeclassifier.NodeMeta{S}
) where {S}
    if !node.is_leaf
        update_using_impurity!(feature_importance, node.l)
        update_using_impurity!(feature_importance, node.r)
        feature_importance[node.feature] +=
            node.node_impurity - node.l.node_impurity - node.r.node_impurity
    end
    return nothing
end

nsample(leaf::Leaf) = length(leaf.values)
nsample(tree::Node) = nsample(tree.left) + nsample(tree.right)
nsample(tree::Root) = nsample(tree.node)

# Numbers of observations for each unique labels
function votes_distribution(labels)
    unique_labels = unique(labels)
    votes = zeros(Int, length(unique_labels))
    @simd for label in labels
        votes[findfirst(==(label), unique_labels)] += 1
    end
    votes
end

function update_pruned_impurity!(
    tree::LeafOrNode{S,T},
    feature_importance::Vector{Float64},
    ntt::Int,
    loss::Function=util.entropy,
) where {S,T}
    all_labels = [tree.left.values; tree.right.values]
    nc = votes_distribution(all_labels)
    nt = length(all_labels)
    ncl = votes_distribution(tree.left.values)
    nl = length(tree.left.values)
    ncr = votes_distribution(tree.right.values)
    nr = nt - nl
    feature_importance[tree.featid] -=
        (nt * loss(nc, nt) - nl * loss(ncl, nl) - nr * loss(ncr, nr)) / ntt
end

function update_pruned_impurity!(
    tree::LeafOrNode{S,T},
    feature_importance::Vector{Float64},
    ntt::Int,
    loss::Function=mean_squared_error,
) where {S,T<:Float64}
    μl = mean(tree.left.values)
    nl = length(tree.left.values)
    μr = mean(tree.right.values)
    nr = length(tree.right.values)
    nt = nl + nr
    μt = (nl * μl + nr * μr) / nt
    feature_importance[tree.featid] -=
        (
            nt * loss([tree.left.values; tree.right.values], repeat([μt], nt)) -
            nl * loss(tree.left.values, repeat([μl], nl)) -
            nr * loss(tree.right.values, repeat([μr], nr))
        ) / ntt
end

################################################################################

function build_stump(
    labels::AbstractVector{T},
    features::AbstractArray{S},
    weights=nothing;
    rng=Random.GLOBAL_RNG,
    impurity_importance::Bool=true,
) where {S,T}
    rng = mk_rng(rng)::Random.AbstractRNG
    t = treeclassifier.fit(;
        X=features,
        Y=labels,
        W=weights,
        loss=treeclassifier.util.zero_one,
        max_features=size(features, 2),
        max_depth=1,
        min_samples_leaf=1,
        min_samples_split=2,
        min_purity_increase=0.0,
        rng,
    )

    return _build_tree(t, labels, size(features, 2), size(features, 1), impurity_importance)
end

function build_tree(
    labels::AbstractVector{T},
    features::AbstractArray{S},
    n_subfeatures=0,
    max_depth=-1,
    min_samples_leaf=1,
    min_samples_split=2,
    min_purity_increase=0.0;
    loss=util.normal_loss::Function,
    rng=Random.GLOBAL_RNG,
    impurity_importance::Bool=true,
) where {S,T}
    if max_depth == -1
        max_depth = typemax(Int)
    end
    if n_subfeatures == 0
        n_subfeatures = size(features, 2)
    end

    rng = mk_rng(rng)::Random.AbstractRNG
    t = treeclassifier.fit(;
        X=features,
        Y=labels,
        W=nothing,
        loss,
        max_features=Int(n_subfeatures),
        max_depth=Int(max_depth),
        min_samples_leaf=Int(min_samples_leaf),
        min_samples_split=Int(min_samples_split),
        min_purity_increase=Float64(min_purity_increase),
        rng,
    )

    return _build_tree(t, labels, size(features, 2), size(features, 1), impurity_importance)
end

function _build_tree(
    tree::treeclassifier.Tree{S,T},
    labels::AbstractVector{T},
    n_features,
    n_samples,
    impurity_importance::Bool,
) where {S,T}
    node = _convert(tree.root, tree.list, labels[tree.labels])
    if !impurity_importance
        return Root{S,T}(node, n_features, Float64[])
    else
        fi = zeros(Float64, n_features)
        update_using_impurity!(fi, tree.root)  # 
        return Root{S,T}(node, n_features, fi ./ n_samples)
    end
end

"""
    prune_tree(tree::Union{Root, LeafOrNode}, purity_thresh=1.0, loss::Function)

Prune `tree` based on prediction accuracy of each node. $DOC_WHATS_A_TREE

* `purity_thresh`: If the prediction accuracy of a stump is larger than this value, the node
  will be pruned and become a leaf.

* `loss`: The loss function for computing node impurity. Available function include
  `DecisionTree_modified.util.entropy`, `DecisionTree_modified.util.gini` and
  `DecisionTree_modified.mean_squared_error`. Defaults are `entropy` and `mean_squared_error` for
  classification tree and regression tree, respectively. If the tree is not a `Root`, this
  argument does not affect the result.

For a tree of type `Root`, when any of its nodes are pruned, the `featim` field will be
updated by recomputing the impurity decrease of that node divided by the total number of
training observations and subtracting the value.  The computation of impurity decrease is
based on node impurity calculated with the loss function provided as the argument
`loss`. The algorithm is as same as that described in the [`impurity_importance`](@ref)
documentation.

This function will recurse until no stumps can be pruned.

!!! warning

    For regression trees, pruning trees based on accuracy may not be an appropriate method.

See also [`build_tree`](@ref).

"""
function prune_tree(
    tree::Union{Root{S,T},LeafOrNode{S,T}},
    purity_thresh=1.0,
    loss::Function=T <: Float64 ? mean_squared_error : util.entropy,
) where {S,T}
    if purity_thresh >= 1.0
        return tree
    end
    ntt = nsample(tree)
    function _prune_run_stump(
        tree::LeafOrNode{S,T}, purity_thresh::Real, fi::Vector{Float64}=Float64[]
    ) where {S,T}
        all_labels = [tree.left.values; tree.right.values]
        majority = majority_vote(all_labels)
        matches = findall(all_labels .== majority)
        purity = length(matches) / length(all_labels)
        if purity >= purity_thresh
            if !isempty(fi)
                update_pruned_impurity!(tree, fi, ntt, loss)
            end
            return Leaf{T}(majority, all_labels)
        else
            return tree
        end
    end
    function _prune_run(tree::Root{S,T}, purity_thresh::Real) where {S,T}
        fi = deepcopy(tree.featim) ## recalculate feature importances
        node = _prune_run(tree.node, purity_thresh, fi)
        return Root{S,T}(node, tree.n_feat, fi)
    end
    function _prune_run(
        tree::LeafOrNode{S,T}, purity_thresh::Real, fi::Vector{Float64}=Float64[]
    ) where {S,T}
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            return _prune_run_stump(tree, purity_thresh, fi)
        else
            left = _prune_run(tree.left, purity_thresh, fi)
            right = _prune_run(tree.right, purity_thresh, fi)
            return Node{S,T}(tree.featid, tree.featval, left, right)
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        pruned = _prune_run(tree, purity_thresh)
    end
    return pruned
end

apply_tree(leaf::Leaf, feature::AbstractVector) = leaf.majority
function apply_tree(tree::Root{S,T}, features::AbstractVector{S}) where {S,T}
    apply_tree(tree.node, features)
end

function apply_tree(tree::Node{S,T}, features::AbstractVector{S}) where {S,T}
    if tree.featid == 0
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::Root{S,T}, features::AbstractMatrix{S}) where {S,T}
    apply_tree(tree.node, features)
end
function apply_tree(tree::LeafOrNode{S,T}, features::AbstractMatrix{S}) where {S,T}
    N = size(features, 1)
    predictions = Array{T}(undef, N)
    for i in 1:N
        predictions[i] = apply_tree(tree, features[i, :])
    end
    if T <: Float64
        return Float64.(predictions)
    else
        return predictions
    end
end

"""
    apply_tree_proba(tree, features, col_labels::AbstractVector)

For the specified `tree`, compute ``P(L=label|X)`` for each row in `features`, returning
an `N_row` x `n_labels` matrix of probabilities, each row summing to one. $DOC_WHATS_A_TREE

`col_labels` is a vector containing the distinct labels, eg. `["versicolor", "virginica",
"setosa"]`. It's order determines the column ordering of the output matrix.

See also [`build_tree`](@ref).

"""
function apply_tree_proba(tree::Root{S,T}, features::AbstractVector{S}, labels) where {S,T}
    apply_tree_proba(tree.node, features, labels)
end
function apply_tree_proba(leaf::Leaf{T}, features::AbstractVector{S}, labels) where {S,T}
    compute_probabilities(labels, leaf.values)
end

function apply_tree_proba(tree::Node{S,T}, features::AbstractVector{S}, labels) where {S,T}
    if tree.featval === nothing
        return apply_tree_proba(tree.left, features, labels)
    elseif features[tree.featid] < tree.featval
        return apply_tree_proba(tree.left, features, labels)
    else
        return apply_tree_proba(tree.right, features, labels)
    end
end
function apply_tree_proba(tree::Root{S,T}, features::AbstractMatrix{S}, labels) where {S,T}
    apply_tree_proba(tree.node, features, labels)
end
function apply_tree_proba(
    tree::LeafOrNode{S,T}, features::AbstractMatrix{S}, labels
) where {S,T}
    stack_function_results(row -> apply_tree_proba(tree, row, labels), features)
end

"""
    build_forest(labels, features, options...; keyword_options...)

Train a random forest model, built on standard CART decision trees, using the specified
`labels` (target) and `features` (patterns). Here:

- `labels` is any `AbstractVector`. If the element type is `Float64`, regression is
  applied, and otherwise classification is applied.

- `features` is any `AbstractMatrix{T}` where `T` supports ordering with `<` (unordered
  categorical features are not supported). The matrix must have size `(n, p)` where `n =
  length(labels)` (observations as rows).

Entropy loss is used for determining node splits, i.e., individual trees are trained by
calling `build_tree` with the `loss=DecisionTree_modified.util.entropy` option.

Use [`apply_forest`](@ref) and [`apply_forest_proba`](@ref) to make predictions on new
features. See the example below.

# Hyperparameters

These are specified as  `options` and `keyword_options`:

## Options

- `n_subfeatures=-1`: number of features to consider at random per split. If equal to
  `-1`, then the square root of the number of features `p` is used

- `n_trees=10`: number of trees to train

- `partial_sampling=0.7`: fraction of samples on which to train each tree

- `max_depth=-1`: maximum depth of the decision trees; no limit if equal to `-1`

- `min_samples_leaf`: the minimum number of samples each leaf needs to have; default is
  `5` for regression and `1` for classification

- `min_samples_split=2`: the minimum number of samples needed for a split

- `min_purity_increase=0`: minimum purity needed to trigger a split

## Keyword options

- `rng=Random.GLOBAL_RNG`: the random number generator or seed (integer) to use. Any
  `AbstractRNG` object supporting `Random.seed!` can be used; each tree gets it's own
  generator.

- `impurity_importance=true`: whether to compute impurity feature importances

# Example

```
features, labels = load_data("iris")    # also see "adult" and "digits" datasets
```

The data loaded are of type `Array{Any}`, so we cast them to concrete types for better
performance:

```
features = float.(features)
labels = string.(labels)
```

Training a random forest classifier using 2 random features, 10 trees, 0.5 portion of
samples per tree, and a maximum tree depth of 6:

```
model = build_forest(labels, features, 2, 10, 0.5, 6)
```

Get predictions on new data:

```
new_features = [5.9 3.0 5.1 1.9
                1.0 2.0 3.9 2.0]
apply_forest(model, new_features)
```

Get probabilistic predictions:
```
apply_forest_proba(
    model,
    new_features,
    ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
)
```

Get impurity feature importances:

```
impurity_importance(model)
```

Run 3-fold cross validation for forests, using 2 random features per split:

```
n_folds=3
n_subfeatures=2
accuracy = nfoldCV_forest(labels, features, n_folds, n_subfeatures)
```

See also [`build_tree`](@ref), [`apply_forest`](@ref), [`apply_forest_proba`](@ref),
[`impurity_importance`](@ref), [`split_importance`](@ref),
[`permutation_importance`](@ref), [`nfoldCV_forest`](@ref).

"""


const ERR_CANT_UPDATE_IMPURITY_IMPORTANCE = DimensionMismatch(
    "Looks like you want to add trees to a model previously trained using a " *
    "different number of features, which means impurity importances " *
    "cannot be updated. Fix by setting `impurity_importance=false`. ",
)
