#!/bin/sh
#SBATCH --job-name="constraint_programming"
#SBATCH --partition=compute
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

module load 2022r1
module load compute
module load openmpi
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mypython

srun python CMain.py "AS" "P" "G" &
srun python CMain.py "AS" "P" "E" &
srun python CMain.py "AS" "P" "O" &
srun python CMain.py "AS" "R" "G" &
srun python CMain.py "AS" "R" "E" &
srun python CMain.py "AS" "R" "O" &
srun python CMain.py "Brute" "P" "G" &
srun python CMain.py "Brute" "P" "E" &
srun python CMain.py "Brute" "P" "O" &
srun python CMain.py "Brute" "R" "G" &
srun python CMain.py "Brute" "R" "E" &
srun python CMain.py "Brute" "R" "O"

