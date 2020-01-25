for seed in 3; do
    #for mode in deterministic; do
    for mode in stochastic deterministic; do
        python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-2 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-2 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-4 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-4 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 100 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-5 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1 --sobolev 0


        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-3 2>&1 | tee 1e-7.log
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-3
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 3e-7 --tikhonov 1e-3 2>&1 | tee 3e-7.log
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 3e-7 --tikhonov 1e-3
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 1e-6 --tikhonov 1e-3 2>&1 | tee 1e-6.log
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 1e-6 --tikhonov 1e-3
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 1e-1 --torsion-penalty 0 --tikhonov 1e-3
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 1e-1 --torsion-penalty 0 --tikhonov 1e-3
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3 2>&1 | tee 0.log
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 10 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-3
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-6
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 1e-6
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 0 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 0 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-6 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 1e-7 --tikhonov 1e-6 --arclength 0 --distance-weight 0.0 --minimum-distance 0.1
        #python3 example2.py      --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 0
        #python3 example2_plot.py --mode $mode --sigma 0.003 --length-scale 0.2 --seed $seed --nsamples 1 --curvature-penalty 0 --torsion-penalty 0 --tikhonov 0
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
