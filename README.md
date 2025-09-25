This repository is the official implementation of [Beyond Greedy: Towards Optimal Deep Classification Trees]

The main loop operates as follows:
 - Find the midpoint and evaluate it to obtain the upper bound.
 - Reduce a circular range (reduction strategy) to obtain two subsets.
 - Recursively apply the process to these subsets.

During evaluation, subtrees of depth are solved recursively and then approximated. Since the approximated method is not optimal, for a trained tree, each node (except the root) can be updated. Therefore, we reoptimize each subtree rooted at these nodes to improve training accuracy.

In the code file, the original Branch-and-Bound method is implemented in each_node.jl. The function bestsplit first splits the data based on the current parameter (before using the evaluate function), then evaluates the node parameters to obtain the upper bound and current cost. Next, delta is computed according to the upper bound, and the search space is partitioned into left_set and right_set, followed by a recursive search of these feasible regions.

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

* mh_cart.jl - function of the moving horizon technique.

* eva.jl - evaluate the misclassification loss of the tree.

* gen_splits.jl - generate the candidate splits of each feature.

* get_ab - get the parameters (a,b) of the tree nodes.

* get_c.jl - get the class labels of the leaf nodes.

* get_data.jl - get the data partitioned to each node.

### Processed Dataset Files - new_data/
All small datasets (59 in total) are available in the folder. 
In the paper, we select 51 datasets from these 59 datasets because some algorithms fail to complete within the time limit.
The excluded datasets include:

26 ***Indian-liver-patient***

33 ***Optical-recognition***

37 ***Qsar-biodegradation***

41 ***Spambase***

45 ***Statlog-project-landsat-satellite***

57 ***Statloglansat***

58 ***Pageblock***

59 ***Pendigits***



The large datasets include ***Avila***, ***Eeg***, ***Skin-segmen-tation***, ***SUSY***, ***HIGGS***, and ***WESAD***, which are publicly accessible from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). For the first three medium-sized datasets, we perform 10 random splits with 75\% of the data used for training and 25\% for testing. For the three larger datasets, we use 2 random splits, also with 75\% for training and 25\% for testing. Due to memory limitations, these datasets are not included in the submission, but we commit to making them publicly available in the future.



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

