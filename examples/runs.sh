for seed in 3; do
    #for mode in deterministic stochastic; do
    for mode in stochastic cvar0.9; do
        mpirun -n 8 python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 200 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 0 --arclength 0 --distance-weight 0 --minimum-distance 0.1 --sobolev 0 2>&1 | tee $mode.log
        python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 200 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 0 --arclength 0 --distance-weight 0 --minimum-distance 0.1 --sobolev 0

        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0

        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0

        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-5 --arclength 0 --distance-weight 10.0 --minimum-distance 0.1 --sobolev 1e-8
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-5 --arclength 0 --distance-weight 10.0 --minimum-distance 0.1 --sobolev 1e-8

        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 1e-8
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 1e-8
    done
done
#for seed in 3; do
#    for mode in stochastic deterministic; do
#        python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 200 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3 2>&1 | tee $mode-$seed-200-pen.log
#        python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 200 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3
#        python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 200 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 0    2>&1 | tee $mode-$seed-200-nopen.log
#        python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 200 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 0
#    done
#done

#for seed in 3; do
#    for mode in stochastic deterministic; do
#        python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 1e-7 --torsion-penalty 1e-7 --tikhonov 1e-3
#        python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 0
#    done
#done
