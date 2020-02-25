rm -rf obfgstasks.txt
touch obfgstasks.txt
for c in 0.1 0.3 1.0; do
    for lam in 0 1e-7 1e-5 1e-3; do
        for lr in 0.1 0.15 0.2; do
            for tau in 10 100 1000; do
                opts="--mode stochastic --sigma 0.003 --length-scale 0.2 --ninsamples 20 --noutsamples 1000 --optimizer sgd --ppp 20 --c $c --lam $lam --lr $lr --tau $tau"
                echo "python3 example2.py $opts; python3 example2_plot.py $opts" >> obfgstasks.txt
            done
        done
    done
done
