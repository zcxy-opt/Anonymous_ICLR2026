# Description: This file is used to achieve an Approximate Branch-and-Reduce Method
using LinearAlgebra
using Printf
using Random, StatsBase, CSV, DataFrames

push!(LOAD_PATH, "./src_cart_cls/")
#using DecisionTree
#user defined modules:
if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
    push!(LOAD_PATH, "./src_cart_cls/")
end
using DecisionTree_modified

include("each_node.jl")
include("gen_splits.jl")
include("get_data.jl")
include("mh_cart.jl")
include("get_c.jl")
include("eva.jl")
include("get_ab.jl")

function loadDataset(dataset, run, dir_path)
    data_path = dir_path * string(dataset) * "_" * string(run) * "_"
    train = DataFrame(CSV.File(data_path * "train", header=false, missingstring="?"))
    train = Matrix(train)
    val = DataFrame(CSV.File(data_path * "val", header=false, missingstring="?"))
    val = Matrix(val)
    test = DataFrame(CSV.File(data_path * "test", header=false, missingstring="?"))
    test = Matrix(test)
    return train, val, test
end

#dir_path = "./new_data/"
dir_path = "./new_data/"

# read args from command line
dataset_start = parse(Int64, ARGS[1])
dataset_end = parse(Int64, ARGS[2])
run_start = parse(Int64, ARGS[3])
run_end = parse(Int64, ARGS[4])
time_limit = parse(Int64, ARGS[5])
tree_depth = parse(Int64, ARGS[6])
epsilon = parse(Float64, ARGS[7]) #terminate condition-epsilon
batch = parse(Float64, ARGS[8])   #mini-batch size
alpha = zeros(100, 10)
#alpha_i = parse.(Float64, readlines("alpha.txt"))

result_train = zeros(100, 12)
result_test = zeros(100, 12)
train_accs = zeros(100, 12)
test_accs = zeros(100, 12)
greedy_accs = zeros(100, 12)
train_times = zeros(100, 12)


for dataset in dataset_start:dataset_end
    println("######## Dataset: ", dataset, " ########")
    for run in pushfirst!(Array(run_start:run_end), 3)
        # try
        start = time()
        Random.seed!(run)
        println("## Dataset: ", dataset, " Run: ", run, " ##")
        train, val, test = loadDataset(dataset, run, dir_path)
        p = size(train, 2) - 1
        X = vcat(train[:, 1:p], val[:, 1:p])
        Y = Int.(vcat(train[:, p+1], val[:, p+1]))
        X_test, Y_test = test[:, 1:p], Int.(test[:, p+1])
        classes = sort(unique(Y))            # get a dataframe with the classes in the data
        class_labels = size(classes, 1)      # number of classes  
        tmp_Y = zeros(Int, size(Y, 1))
        tmp_Y_test = zeros(Int, size(Y_test, 1))
        for i in 1:class_labels
            tmp_Y[findall(x -> x == classes[i], Y)] .= i
            tmp_Y_test[findall(x -> x == classes[i], Y_test)] .= i
        end
        Y = tmp_Y
        Y_test = tmp_Y_test
        classes = sort(unique(Y))           # get a dataframe with the classes in the data
        class_labels = size(unique(Y), 1)   # number of classes  
        if run == 1
            println("dataset: ", dataset, " run: ", run, " n_train: ", size(X, 1), " n: ", size(vcat(X, X_test), 1), " p: ", p, " class_labels: ", size(unique(Y), 1))
        end
        test_run = Inf
        alpha_best = 0
        opt_train = 0
        opt_test = 0
        for alpha_i in 0.0:0.05:0               #alpha_i = 0.0:0.05:1.0: alphg tuning parameters for CART
            A, B, C, _ = mh_cart(X, Y, tree_depth, epsilon, alpha_i, batch)
            println(" A: ", A, " B: ", B, " C: ", C)
            train = eva(X, Y, A, B, C)
            test = eva(X_test, Y_test, A, B, C)
            if test < test_run
                test_run = test
                alpha_best = alpha_i
                opt_train = train
                opt_test = test
                println("alpha: ", alpha_best, " test: ", test_run)
            end
        end
        alpha[dataset, run] = alpha_best
        train_acc = 1 - opt_train / size(X, 1)
        test_acc = 1 - opt_test / size(X_test, 1)
        train_acc = train_acc * 100
        test_acc = test_acc * 100
        train_accs[dataset, run] = train_acc
        test_accs[dataset, run] = test_acc
        end_time = time()
        train_times[dataset, run] = end_time - start
        println("Train Acc: ", train_acc, " Train_cost ", opt_train, " Test Acc: ", test_acc, " Time: ", end_time - start)
        result_train[dataset, run] = train_acc
        result_test[dataset, run] = test_acc
        # catch e
        #     println("Error in dataset: ", dataset, " run: ", run, " error is")
        #     showerror(stdout, e)
        #     train_accs[dataset, run] = 0
        #     test_accs[dataset, run] = 0
        #     train_times[dataset, run] = 0
        # end # try

        CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_training_accuracys_updated.csv", DataFrame(train_accs, :auto))
        CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_test_accuracys_updated.csv", DataFrame(test_accs, :auto))
        CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_training_times_updated.csv", DataFrame(train_times, :auto))
    end # run
end # dataset
# std and mean of the results
train_accs[:, 11] = mean(train_accs[:, 1:10], dims=2)
train_accs[:, 12] = std(train_accs[:, 1:10], dims=2)
test_accs[:, 11] = mean(test_accs[:, 1:10], dims=2)
test_accs[:, 12] = std(test_accs[:, 1:10], dims=2)
train_times[:, 11] = mean(train_times[:, 1:10], dims=2)
train_times[:, 12] = std(train_times[:, 1:10], dims=2)
println("Mean Train Acc: ", train_accs[:, 11])
println("Mean Test Acc: ", test_accs[:, 11])
println("Mean Train Time: ", train_times[:, 11])
result_train[:, 11] = mean(result_train[:, 1:10], dims=2)
result_test[:, 12] = mean(result_test[:, 1:10], dims=2)
CSV.write("Train_Result", DataFrame(result_train, :auto))
CSV.write("Train_Test", DataFrame(result_test, :auto))
# save the results
CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_training_accuracys_updated.csv", DataFrame(train_accs, :auto))
CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_test_accuracys_updated.csv", DataFrame(test_accs, :auto))
CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_training_times_updated.csv", DataFrame(train_times, :auto))


