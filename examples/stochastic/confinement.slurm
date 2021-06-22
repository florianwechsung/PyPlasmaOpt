#!/bin/bash
#
#SBATCH --job-name=stochastic
#SBATCH --output=logs/slurm_conf_%A_%a.out
#SBATCH --error=logs/slurm_conf_%A_%a.err
#SBATCH -p cs
#SBATCH --time=24:00:00
#SBATCH --mem=170GB

#SBATCH --nodes=1
#SBATCH --tasks=48

##SBATCH --nodes=2
##SBATCH --tasks=96

##SBATCH --nodes=4
##SBATCH --tasks=192

##SBATCH --nodes=6
##SBATCH --tasks=192

##SBATCH --nodes=8
##SBATCH --tasks=384

#SBATCH --tasks-per-node=48
#SBATCH --cpus-per-task=1
##SBATCH --dependency=afterok:887288

 
module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=1

cd /scratch/fw18/PyPlasmaOpt/examples/stochastic
source /scratch/fw18/plasmaenv/bin/activate


# run with sbatch --array=0-7 confinement.slurm

IG=${SLURM_ARRAY_TASK_ID}
  for EN in 250 500 1000; do



    #case="output-greene/mc13_config-ncsx_mode-deterministic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-0_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-${IG}_ip-l2_optim-scipy/"
    case="output-greene/mc13_config-ncsx_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-1024_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-${IG}_ip-l2_optim-scipy/"

    mpiexec python3 confinement_stochastic.py --case $case           --energy $EN  --tmax 1e-2 --nparticles 20 --it -1
    if [[ $IG -eq 0 ]]; then
      mpiexec python3 confinement_stochastic.py --case $case           --energy $EN  --tmax 1e-2 --nparticles 20 --it 0
    fi
  done
#done
