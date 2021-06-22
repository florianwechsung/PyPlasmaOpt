import matplotlib.pyplot as plt
import numpy as np
from pyplasmaopt import *
from mayavi import mlab
import os
os.makedirs('riskaverse', exist_ok=True)
def get_dir(n, ig, ntcoils, mode):
    ppp = 20 if ntcoils < 10 else 15
    return f"output-greene/mc13_config-ncsx_mode-{mode}_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-{n}_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/"


print("Configuration".ljust(40), "Mean".ljust(22), "CVaR(0.5)".ljust(10), "CVaR(0.9)".ljust(10), "CVaR(0.95)".ljust(10), "CVaR(0.99)".ljust(10))
for ig in range(8):
    for mode in ['stochastic', 'cvar0p9', 'cvar0p95']:
        try:
            vals = np.load(get_dir(1024, ig, 6, mode) + "Jvals_outofsample.npy")
        except:
            continue
        mean = np.mean(vals)
        std = np.std(vals)
        err = std/np.sqrt(len(vals))
        label = f"{mode} IG={ig}"
        var50 = np.quantile(vals, 0.5)
        cvar50 = np.mean(vals[vals>=var50])
        var90 = np.quantile(vals, 0.9)
        cvar90 = np.mean(vals[vals>=var90])
        var95 = np.quantile(vals, 0.95)
        cvar95 = np.mean(vals[vals>=var95])
        var99 = np.quantile(vals, 0.99)
        cvar99 = np.mean(vals[vals>=var99])
        # print(("%s:" % label).ljust(40), ("[%.3e, %.3e]" % (mean-err, mean+err)).ljust(22), ("%.3e" % np.mean(cvar50)).ljust(10), ("%.3e" % np.mean(cvar90)).ljust(10), ("%.3e" % np.mean(cvar95)).ljust(10), ("%.3e" % np.mean(cvar99)).ljust(10))
        print(("%s:" % label).ljust(40), ("%.3e" % (mean)).ljust(22), ("%.3e" % np.mean(cvar50)).ljust(10), ("%.3e" % np.mean(cvar90)).ljust(10), ("%.3e" % np.mean(cvar95)).ljust(10), ("%.3e" % np.mean(cvar99)).ljust(10))
for ig in range(8):
    for mode in ['stochastic', 'cvar0p9', 'cvar0p95']:
        try:
            vals = np.load(get_dir(1024, ig, 6, mode) + "Jvals_insample.npy")
        except:
            continue
        mean = np.mean(vals)
        std = np.std(vals)
        err = std/np.sqrt(len(vals))
        label = f"{mode} IG={ig}"
        var50 = np.quantile(vals, 0.5)
        cvar50 = np.mean(vals[vals>=var50])
        var90 = np.quantile(vals, 0.9)
        cvar90 = np.mean(vals[vals>=var90])
        var95 = np.quantile(vals, 0.95)
        cvar95 = np.mean(vals[vals>=var95])
        var99 = np.quantile(vals, 0.99)
        cvar99 = np.mean(vals[vals>=var99])
        # print(("%s:" % label).ljust(40), ("[%.3e, %.3e]" % (mean-err, mean+err)).ljust(22), ("%.3e" % np.mean(cvar50)).ljust(10), ("%.3e" % np.mean(cvar90)).ljust(10), ("%.3e" % np.mean(cvar95)).ljust(10), ("%.3e" % np.mean(cvar99)).ljust(10))
        print(("%s:" % label).ljust(40), ("%.3e" % (mean)).ljust(22), ("%.3e" % np.mean(cvar50)).ljust(10), ("%.3e" % np.mean(cvar90)).ljust(10), ("%.3e" % np.mean(cvar95)).ljust(10), ("%.3e" % np.mean(cvar99)).ljust(10))

import seaborn as sns
import pandas as pd
plt.figure(figsize=(9,5))
sns.set_style('whitegrid')

labels_density = []
data_density = []
ax = plt.gca()
for mode in ['stochastic', 'cvar0p9']:
    color = next(ax._get_lines.prop_cycler)['color']
    for ig in range(8):
        try:
            data = np.load(get_dir(1024, ig, 6, mode) + "Jvals_outofsample.npy")
            # data = np.load(get_dir(1024, ig, 6, mode) + "Jvals_insample.npy")
        except:
            continue
        label = mode if ig == 0 else None
        p = sns.kdeplot(data, color=color, label=label)
        d = p.get_lines()[-1].get_data() 
        data_density += d
        labels_density += [f"x_mode_{mode}_ig_{ig}", f"y_mode_{mode}_ig_{ig}"]

plt.xlabel('Objective value')
plt.ylabel('pdf')
plt.legend()
plt.title('Probability density at optimal configuration')
plt.xlim((0, 0.015))
# plt.ylim((0, 1.))
plt.savefig(f"riskaverse_dist.pdf", dpi=600, bbox_inches='tight')
np.savetxt("riskaverse/riskaverse_density.txt", np.asarray(data_density).T, delimiter=";", header=";".join(labels_density), comments="")
plt.close()
# plt.show()
plt.figure(figsize=(9,5))
labels_cvar = []
data_cvar = []
for mode in ['stochastic', 'cvar0p9']:
    color = next(ax._get_lines.prop_cycler)['color']
    for ig in range(8):
        try:
            data = np.load(get_dir(1024, ig, 6, mode) + "Jvals_outofsample.npy")
            # data = np.load(get_dir(1024, ig, 6, mode) + "Jvals_insample.npy")
        except:
            continue
        data = np.flip(np.sort(data))
        ones = np.ones(data.shape)
        cvars = np.cumsum(data)/np.cumsum(ones)
        label = mode if ig == 0 else None
        x = 1-np.cumsum(ones)/len(ones)
        y = cvars
        plt.plot(x, y, color=color, label=label)
        labels_cvar += [f"x_mode_{mode}_ig_{ig}", f"y_mode_{mode}_ig_{ig}"]
        data_cvar += [x, y]
np.savetxt("riskaverse/riskaverse_cvar.txt", np.asarray(data_cvar).T[::1024, :], delimiter=";", header=";".join(labels_cvar), comments="")
plt.ylim((0.005, 0.01))
plt.xlabel('alpha')
plt.ylabel('CVaR_alpha')
plt.legend()
plt.savefig(f"riskaverse/riskaverse_cvar.pdf", dpi=600, bbox_inches='tight')
plt.close()
# plt.show()
