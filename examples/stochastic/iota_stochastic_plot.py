import numpy as np
import matplotlib.pyplot as plt


sigma = "0p01"
sigma = "0p003"
def get_dir(n, ig, ntcoils):
    ppp = 20 if ntcoils < 10 else 15
    if n == 1:
        return f"output-greene/mc13_config-ncsx_mode-deterministic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-0_noutsamples-1024_seed-0_sigma-{sigma}_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/iotas/"
    else:
        return f"output-greene/mc13_config-ncsx_mode-stochastic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-{n}_noutsamples-1024_seed-0_sigma-{sigma}_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/iotas/"



target_iota = 0.395938929522566

import seaborn as sns
import pandas as pd
plt.figure(figsize=(5,5))
sns.set_style('whitegrid')

labels_density = []
data_density = []
ax = plt.gca()
for order in [6]:
    for n in [1, 1024]:
        color = next(ax._get_lines.prop_cycler)['color']
        for ig in [0, 1, 2, 3, 4, 5, 6, 7]:
            outdir = get_dir(n, ig, order)
            try:
                iotas = np.load(outdir + "/iotas.npy")
                # p = sns.kdeplot(np.abs(iotas-target_iota), color=color)
                p = sns.kdeplot(iotas, color=color)
                d = p.get_lines()[-1].get_data() 
                data_density += d
                labels_density += [f"x_n_{n}_order_{order}_ig_{ig}", f"y_n_{n}_order_{order}_ig_{ig}"]
            except Exception as ex:
                print(ex)
                pass

import os
os.makedirs('iotas', exist_ok=True)
np.savetxt(f"iotas/iota_density_{sigma}.txt", np.asarray(data_density).T, delimiter=";", header=";".join(labels_density), comments="")
plt.show()
