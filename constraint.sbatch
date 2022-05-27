#!/bin/sh
#SBATCH --job-name="constraint_programming"
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=3G
#SBATCH --mail-type=END

module load 2022r1
module load compute
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mypython

srun python CMain.py
