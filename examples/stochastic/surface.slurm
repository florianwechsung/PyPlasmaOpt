#!/bin/bash
#


### Run with sbatch --array=1-64 surface.slurm

#SBATCH --job-name=stochastic
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
##SBATCH -p cs
#SBATCH --time=24:00:00
#SBATCH --mem=4GB

#SBATCH --nodes=1
#SBATCH --tasks=1

#SBATCH --tasks-per-node=1

 
module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5

cd /scratch/fw18/PyPlasmaOpt/examples/stochastic
source /scratch/fw18/plasmaenv/bin/activate

export OMP_NUM_THREADS=1
lineno=${SLURM_ARRAY_TASK_ID}
echo $lineno
line=`cat tasks.txt | head -${lineno} | tail -1`
echo $line
mpiexec -n 1 $line
