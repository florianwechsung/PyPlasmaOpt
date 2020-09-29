python3 example3_simple.py --ppp 20 --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 0.    --torsion 0.0  --curvature 0.00 --arclength 0.00
python3 example3_plot.py   --ppp 20 --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 0.    --torsion 0.0  --curvature 0.00 --arclength 0.00

python3 example3_simple.py --ppp 20 --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 1000.    --torsion 0.0  --curvature 0.00 --arclength 0.00
python3 example3_plot.py   --ppp 20 --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 1000.    --torsion 0.0  --curvature 0.00 --arclength 0.00

python3 example3_simple.py --ppp 20 --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 1000.    --torsion 0.0  --curvature 1e-6 --arclength 0.00
python3 example3_plot.py   --ppp 20 --at-optimum --output ncsx --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 1000.    --torsion 0.0  --curvature 1e-6 --arclength 0.00
