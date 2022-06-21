# Genetic Algorithm to Evolve a Search Procedure for Program Synthesis
This branch contains the part of code responsible for evolving the search procedure for program synthesis. It was created by Michał Okoń as a part 
of the bachelor thesis titled 'Evolving a Search Procedure for Program Synthesis'.

The genetic algorithm is located in the metasynthesis/search_procedure/search_synthesizer.py and can be run by simply running this python file.
Parameters can be modified, according to the comments left in the code.

In the root directory, there are mutliple scripts that allow to run experiments. 
###analyze_main.py
Runs the analysis of execution time of single search procedures.
###run_single.py
Runs single search procedures (including the combined search) and analyzes their outcome.
###searchmain.py
Runs the seearch procedure evolution and plots the most important statistics.
### files with .sbatch extension
Those files are used to run each of the scripts listed above on the DelftBlue cluster.

# BEP_project_synthesis
The code in this branch is based on the code made by F. Azimzade, B. Jenneboer, N. Matulewicz, S. Rasing and V. van 
Wieringen as a  part of the BEP projects.

## Source of training/test data
Robot and Pixel test/training data was generated by A. Cropper and S. Dumančić for their paper "Learning large logic programs by going beyond entailment." arXiv preprint arXiv:2004.09855 (2020).

The String test/training data was received from S. Dumančić who took them from the paper by Lin, Dianhuan, et al. "Bias reformulation for one-shot function induction." (2014).

## Running the BEP project

There are three runnable main files for this project: main.py, debug_main.py and hpc_main.py.

### main.py
This main is an easy to run file that can run test cases and print out some results. In this file you can set the time limit, search algorithm, domain, heuristic (or distance measure) to use and number of trials that will be run. In the file each setting is explained further.

### debug_main.py
This main implements an easy way to debug the code. The settings are almost the same as in the normal main.py file, only without the number of trials. There is a more elaborate way to set which test cases to run, which is explained in the file. When a test case is completed (either solved or time out) a console line is printed with its results.

### hpc_main.py
This main can be used for the High Performance Cluster (HPC). With system arguments the search algorithm, domain and distance heuristic can be set. You can also run this on your own PC. Test case results will be stored when run by this file.

