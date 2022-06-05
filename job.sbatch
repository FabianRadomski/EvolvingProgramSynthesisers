#!/bin/sh
#SBATCH --partition=compute
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000

module use /opt/insy/modulefiles
module load miniconda/3.9

srun python main.py