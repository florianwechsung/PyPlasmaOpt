rm -rf tasks.txt
touch tasks.txt
#python3 confinement_stochastic.py --ppp 20 --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 0.    --torsion 0.0  --curvature 0.00 --arclength 0.00
#gtime -v python3 -m cProfile -o profile.stat optimisation_stochastic.py --ppp 10 --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --ninsamples 32 --noutsamples 32 --seed 1
export OMP_NUM_THREADS=1
export MPI_SIZE=16

#NINSAMPLES=32
#NOUTSAMPLES=32
#PPP=20
#gtime -v mpiexec -n 8 python3                 optimisation_stochastic.py --mode stochastic --ppp $PPP --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --ninsamples $NINSAMPLES --noutsamples $NOUTSAMPLES --seed 1 --sigma 0.001 --length-scale 0.2
#gtime -v mpiexec -n 8 python3                 optimisation_stochastic.py --mode stochastic --ppp $PPP --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --ninsamples $NINSAMPLES --noutsamples $NOUTSAMPLES --seed 1 --sigma 0.003 --length-scale 0.2
#gtime -v mpiexec -n 8 python3                 optimisation_stochastic.py --mode stochastic --ppp $PPP --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --ninsamples $NINSAMPLES --noutsamples $NOUTSAMPLES --seed 1 --sigma 0.010 --length-scale 0.2


NOUTSAMPLES=1024
PPP=20
SIGMA=0.01
echo "Increase sample size"
task="gtime -v mpiexec -n $MPI_SIZE python3 optimisation_stochastic.py --mode deterministic --ppp $PPP --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --ninsamples 0 --noutsamples $NOUTSAMPLES --seed 1 --sigma $SIGMA --length-scale 0.2"
echo $task >> tasks.txt
##for ninsamples in 16 64 256 1024; do
for ninsamples in 16 1024; do
    task="gtime -v mpiexec -n $MPI_SIZE python3 optimisation_stochastic.py --mode stochastic --ppp $PPP --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --ninsamples $ninsamples --noutsamples $NOUTSAMPLES --seed 1 --sigma $SIGMA --length-scale 0.2"
    echo $task >> tasks.txt
done

task="gtime -v mpiexec -n $MPI_SIZE python3 optimisation_stochastic.py --mode cvar0.95 --ppp $PPP --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --ninsamples 1024 --noutsamples $NOUTSAMPLES --seed 1 --sigma $SIGMA --length-scale 0.2"
echo $task >> tasks.txt
task="gtime -v mpiexec -n $MPI_SIZE python3 optimisation_stochastic.py --mode cvar0.99 --ppp $PPP --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --ninsamples 1024 --noutsamples $NOUTSAMPLES --seed 1 --sigma $SIGMA --length-scale 0.2"
echo $task >> tasks.txt
