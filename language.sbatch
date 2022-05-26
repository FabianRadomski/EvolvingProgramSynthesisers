#!/bin/sh
#SBATCH --job-name="genetic_language"
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=4G
#SBATCH --mail-type=END

module load 2022r1
module load compute
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mypython
python run_genetic.py