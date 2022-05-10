#!/bin/sh
#SBATCH --job-name="constraint_programming"
#SBATCH --partition=compute
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --mem=3G
#SBATCH --mail-type=END

module load 2022r1
module load compute
module load miniconda
conda create -n mypython python=3.9
srun conda CMain.py
