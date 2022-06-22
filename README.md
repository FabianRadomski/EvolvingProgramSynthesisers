### Running the genetic algorithm
A single run of the genetic algorithm evolves a language for a single domain. This can be done by running language_run_genetic.py. In the run_genetic_algorithm_once function, you can change the desired parameters. This runs an instance of the genetic algorithm found in: metasynthesis/programming_language/evolve_language.py. 

### Experiments
The experiments can also be run from language_run_genetic.py. You can do this by uncommenting the desired experiment. Tuning the parameters with regards to these experiments can be done in metasynthesis/programming_language/plot_statistics.py.

### Supercomputer
The file language.sbatch can be used to run the language_run_genetic.py on a supercomputer. This is done by transferring the relevant directories to a supercomputer. Then, loggin in on the supercomputer and setting up a python environment with the required modules. Finally, the language.sbatch file can be ran using sbatch language.sbatch (assuming the supercomputer setup is comparable to the DelftBlue supercomputer). 

### Acknowledgement codebase

Part of this codebase stems from the BEP projects from F. Azimzade, B. Jenneboer, N. Matulewicz, S. Rasing and V. van Wieringen. This concerns the basic implementations for everything other than the metasynthesis directory and related files to run parts of that directory. 

### Acknowledgement training/test data
Robot and Pixel test/training data was generated by A. Cropper and S. Dumančić for their paper "Learning large logic programs by going beyond entailment." arXiv preprint arXiv:2004.09855 (2020).

The String test/training data was received from S. Dumančić who took them from the paper by Lin, Dianhuan, et al. "Bias reformulation for one-shot function induction." (2014).
