#!/bin/sh
#SBATCH --job-name="evolving_search"
#SBATCH --partition=compute
#SBATCH --time=10:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=3G
#SBATCH --mail-type=END

module load 2022r1
module load compute
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /beegfs/home/mokon/searchenv
python searchmain.py