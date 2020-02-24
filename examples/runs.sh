if [ "${HOSTNAME: -7}" = "nyu.edu" ]; then 
    mympi='mpirun -n 72 -hosts voyager2,voyager3,voyager4'
    export OMP_NUM_THREADS=1
    export MESA_GL_VERSION_OVERRIDE=4.5
    mypython='xvfb-run -s "-screen 0 1920x1080x24" python3.7'
else
    mympi="mpirun -n 8"
    export OMP_NUM_THREADS=2
    mypython='python3'
fi



for seed in 3; do
    for mode in stochastic cvar0.9 deterministic; do
        for nsamples in 200 2000; do
            eval ${mympi} ${mypython} example2.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --ninsamples $nsamples --noutsamples 20000 --curvature-pen 0 --torsion-pen 0 --tikhonov 0 --arclength 0 --dist-weight 0 --min-dist 0.1 --sobolev 0
            eval ${mypython} example2_plot.py     --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --ninsamples $nsamples --noutsamples 20000 --curvature-pen 0 --torsion-pen 0 --tikhonov 0 --arclength 0 --dist-weight 0 --min-dist 0.1 --sobolev 0

            eval ${mympi} ${mypython} example2.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --ninsamples $nsamples --noutsamples 20000 --curvature-pen 0 --torsion-pen 0 --tikhonov 1e-5 --arclength 0 --dist-weight 1 --min-dist 0.1 --sobolev 0
            eval ${mypython} example2_plot.py     --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --ninsamples $nsamples --noutsamples 20000 --curvature-pen 0 --torsion-pen 0 --tikhonov 1e-5 --arclength 0 --dist-weight 1 --min-dist 0.1 --sobolev 0
        done
    done
done


#for ninsamples in 8 16 32 64; do
#    for optimizer in bfgs sgd; do
#        mpirun -n 8 python3 example2.py --mode stochastic --ninsamples $ninsamples --noutsamples 320 --ppp 20 --optimizer $optimizer
#    done
#done
