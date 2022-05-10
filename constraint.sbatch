#!/bin/sh
#SBATCH --job-name="constraint_programming"
#SBATCH --partition=compute
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --mem=3G
#SBATCH --mail-type=END

module load 2022r1
module load intel/oneapi-all

export I_MPI_PMI_LIBRARY=/cm/shared/apps/slurm/current/lib64/libpmi2.so


module use /opt/insy/modulefiles
module load miniconda/3.9

srun conda CMain.py
