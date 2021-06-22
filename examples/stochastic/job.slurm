#!/bin/bash
#
#SBATCH --job-name=stochastic
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH -p cs
#SBATCH --time=24:00:00
#SBATCH --mem=40GB

##SBATCH --nodes=1
##SBATCH --tasks=48

##SBATCH --nodes=2
##SBATCH --tasks=96

##SBATCH --nodes=4
##SBATCH --tasks=192

#SBATCH --nodes=6
#SBATCH --tasks=192

##SBATCH --nodes=8
##SBATCH --tasks=384

#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
##SBATCH --dependency=afterok:887288

 
module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=1

cd /scratch/fw18/PyPlasmaOpt/examples/stochastic
source /scratch/fw18/plasmaenv/bin/activate



# run with sbatch --array=0-7 job.slurm

### # Monte Carlo Convergence 
### CTR="mc13"
### IG=$SLURM_ARRAY_TASK_ID
### SIGM=0.003
### DISTR=gaussian
### PPP=20
### TIK=0
### IP="l2"
### #IP="Hk"
### CARCLEN=1e-4
### CLEN=1
### CONFIG="ncsx"
### SOBOLEV=0
### #for NINSAMPLES in 4 16 64 256 1024; do
### #	mpiexec python3 optimisation_stochastic.py --ppp ${PPP} --config $CONFIG --output ${CTR} --Nt_ma 4 --Nt_coils 6 --ninsamples ${NINSAMPLES} --noutsamples 1024 --seed 0 \
### #		--ip ${IP} --sigma ${SIGM} --length-scale 0.2 --distribution ${DISTR} --ig ${IG} --tikhonov ${TIK} --opt scipy --mode stochastic --carclen ${CARCLEN} --clen ${CLEN} --curvature 1e-6 --distw 1000 --torsion 0 --sobolev ${SOBOLEV}
### #done
### mpiexec python3 optimisation_stochastic.py --ppp ${PPP} --config $CONFIG --output ${CTR} --Nt_ma 4 --Nt_coils 6 --ninsamples 0 --noutsamples 1024 --seed 0 \
### 	--ip ${IP} --sigma ${SIGM} --length-scale 0.2 --distribution ${DISTR} --ig ${IG} --tikhonov ${TIK} --opt scipy --mode deterministic --carclen ${CARCLEN} --clen ${CLEN} --curvature 1e-6 --distw 1000 --torsion 0 --sobolev ${SOBOLEV}


# CVAR
CTR="mc13"
IG=$SLURM_ARRAY_TASK_ID
SIGM=0.01
DISTR=gaussian
PPP=20
TIK=0
IP="l2"
#IP="Hk"
CARCLEN=1e-4
CLEN=1
CONFIG="ncsx"
SOBOLEV=0
NINSAMPLES=1024
mpiexec python3 optimisation_stochastic.py --ppp ${PPP} --config $CONFIG --output ${CTR} --Nt_ma 4 --Nt_coils 6 --ninsamples ${NINSAMPLES} --noutsamples 1024 --seed 0 \
	--ip ${IP} --sigma ${SIGM} --length-scale 0.2 --distribution ${DISTR} --ig ${IG} --tikhonov ${TIK} --opt scipy --mode cvar0.9 --carclen ${CARCLEN} --clen ${CLEN} --curvature 1e-6 --distw 1000 --torsion 0 --sobolev ${SOBOLEV}
mpiexec python3 optimisation_stochastic.py --ppp ${PPP} --config $CONFIG --output ${CTR} --Nt_ma 4 --Nt_coils 6 --ninsamples ${NINSAMPLES} --noutsamples 1024 --seed 0 \
	--ip ${IP} --sigma ${SIGM} --length-scale 0.2 --distribution ${DISTR} --ig ${IG} --tikhonov ${TIK} --opt scipy --mode cvar0.95 --carclen ${CARCLEN} --clen ${CLEN} --curvature 1e-6 --distw 1000 --torsion 0 --sobolev ${SOBOLEV}
mpiexec python3 optimisation_stochastic.py --ppp ${PPP} --config $CONFIG --output ${CTR} --Nt_ma 4 --Nt_coils 6 --ninsamples ${NINSAMPLES} --noutsamples 1024 --seed 0 \
	--ip ${IP} --sigma ${SIGM} --length-scale 0.2 --distribution ${DISTR} --ig ${IG} --tikhonov ${TIK} --opt scipy --mode cvar0.99 --carclen ${CARCLEN} --clen ${CLEN} --curvature 1e-6 --distw 1000 --torsion 0 --sobolev ${SOBOLEV}

