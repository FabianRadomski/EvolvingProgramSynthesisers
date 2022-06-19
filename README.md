# BEP_project_synthesis

This code is part of the BEP projects of F. Azimzade, B. Jenneboer, N. Matulewicz, S. Rasing and V. van Wieringen.

## Source of training/test data
Robot and Pixel test/training data was generated by A. Cropper and S. Dumančić for their paper "Learning large logic programs by going beyond entailment." arXiv preprint arXiv:2004.09855 (2020).

The String test/training data was received from S. Dumančić who took them from the paper by Lin, Dianhuan, et al. "Bias reformulation for one-shot function induction." (2014).

## Running the project

There are three runnable main files for this project: main.py, debug_main.py and hpc_main.py.

### main.py
This main is an easy to run file that can run test cases and print out some results. In this file you can set the time limit, search algorithm, domain, heuristic (or distance measure) to use and number of trials that will be run. In the file each setting is explained further.

### debug_main.py
This main implements an easy way to debug the code. The settings are almost the same as in the normal main.py file, only without the number of trials. There is a more elaborate way to set which test cases to run, which is explained in the file. When a test case is completed (either solved or time out) a console line is printed with its results.

### hpc_main.py
This main can be used for the High Performance Cluster (HPC). With system arguments the search algorithm, domain and distance heuristic can be set. You can also run this on your own PC. Test case results will be stored when run by this file.

### objective_function_experiment.py
This file contains the code for running experiments with the Genetic Algorithm evolving objective functions (metasynthesis/performance_function/evolving_function.py).
Specifically, the following parameters are needed for the 'run_experiment' method:
- domain ('string' or 'robot')
- num_generations (number of generations)
- pop_size (population size)
- p_c (crossover probability)
- p_m (mutation probability)
- tournament_size
- elite_size
- the weights w1, w2, w3 for the fitness function
- d_max (maximum height of a tree in the initial population)
- pr_op (probability of placing an operator node in the construction of a random tree)
- time_out (timeout given to Brute)
The results are written to a file.

### Results for GeneticObjective
The results that are reported in the paper 'Genetic Algorithm for Evolving an Objective Function of a Program Synthesizer' can be found under obj_fun_experiments/.