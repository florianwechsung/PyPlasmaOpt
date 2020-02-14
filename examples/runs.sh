mympi="mpirun -n 8"
export OMP_NUM_THREADS=2
mypython='python3'

for seed in 3; do
    for mode in deterministic stochastic cvar0.9; do
        for nsamples in 200 2000; do
            eval ${mympi} ${mypython} example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --ninsamples $nsamples --noutsamples 20000 --curvature-pen 0 --torsion-pen 0 --tikhonov 1e-5 --arclength 0 --dist-weight 1 --min-dist 0.1 --sobolev 0
            eval ${mypython} example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --ninsamples $nsamples --noutsamples 20000 --curvature-pen 0 --torsion-pen 0 --tikhonov 1e-5 --arclength 0 --dist-weight 1 --min-dist 0.1 --sobolev 0

            eval ${mympi} ${mypython} example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples $nsamples --noutsamples 20000 --curvature-pen 0 --torsion-pen 0 --tikhonov 1e-5 --arclength 0 --dist-weight 0.0 --min-dist 0.1 --sobolev 0
            eval ${mypython} example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples $nsamples --noutsamples 20000 --curvature-pen 0 --torsion-pen 0 --tikhonov 1e-5 --arclength 0 --dist-weight 0.0 --min-dist 0.1 --sobolev 0
        done
    done
done


#for ninsamples in 8 16 32 64; do
#    for optimizer in bfgs sgd; do
#        mpirun -n 8 python3 example2.py --mode stochastic --ninsamples $ninsamples --noutsamples 320 --ppp 20 --optimizer $optimizer
#    done
#done
