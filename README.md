This repository is the official implementation of [A Moving-Horizon Approximate Branch-and-Reduce Method for Training Deep Classification Trees]

## Requirements
* Julia v1.11.2
* Julia Packages
    * LinearAlgebra
    * CSV v0.10.4
    * DataFrames v1.3.4
    * Plots v1.31.2
    * StatsBase v0.33.18
    * LinearAlgebra
    * Printf
    * Random

## File list
###  Source Files - src/
* test_mh.jl - main function.

* each_node.jl - function of the branch and reduce method.

* mh_cart.jl - funtion of the moving horizon technique.

* eva.jl - evaluate the misclassigication loss of the tree.

* gen_splits.jl - generate the candidate splits of each feature.

* get_ab - get the parameters (a,b) of the tree nodes.

* get_c.jl - get the class labels of the leaf nodes.

* get_data.jl - get the data partitioned to each node.

### Dataset Files - new_data/
All small datasets (59 in total) are available in the folder. The large datasets include
***Avila***, ***Eeg***, ***Skin-segmen-tation***, ***SUSY***, ***HIGGS***, and ***WESAD***, which can be accessed from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) . In this folder, each dataset was randomly split into 10 runs, with 75\% for training and 25\% for testing, respectively.

## Run the file
 * run.sh - run this project.

A basic command is:
 > julia test_mh.jl 1 59 1 10 10 4 0 1 

This code takes eight inputs: 
* args[1] the index of the first dataset 
* args[2] the index of the end dataset 
* args[3] the index of the first run 
* args[4] the index of the end run 
* args[5] time limit for each run
* args[6] tree depth
* args[7] epsilon
* args[8] batch size

