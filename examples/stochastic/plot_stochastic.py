from pyplasmaopt import *
import itertools
from math import log10, floor, ceil
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)




outdirs = [
    "output-greene/ncsx_mc2_atopt_mode-deterministic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-0_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-0p0_ig-0_ip-l2_optim-scipy/",
    "output-greene/ncsx_mc2_atopt_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-0p0_ig-0_ip-l2_optim-scipy/",
    "output-greene/ncsx_mc2_atopt_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-0p0_ig-0_ip-l2_optim-scipy/",
]

labels = [
    "Deterministic",
    "Stochastic (n=4)",
    "Stochastic (n=4096)",
]



plt.figure(figsize=(9,5))
ax = plt.gca()
for i, (outdir, label) in enumerate(list(zip(outdirs, labels))[j] for j in range(3)):
    try:
        colors = next(ax._get_lines.prop_cycler)['color']
        Jvals = np.load(outdir + "Jvals.npy")
        plt.semilogy(Jvals, "--", label=label + " - J", color=colors, linewidth=2)
        # Jvals_no_noise = np.load(outdir + "Jvals_no_noise.npy")
        # plt.semilogy(Jvals_no_noise, ":", label=label + " - J", color=colors, linewidth=2)
        # out_of_sample_means = np.load(outdir + "out_of_sample_means.npy")
        # plt.semilogy(out_of_sample_means, "-", label=label + " - OoS Mean", color=colors, linewidth=1.5)
    except: pass
plt.grid(which='both')
plt.xlabel('Iteration')
# plt.title('Objective value and Out-of-Sample mean')
plt.ylim((1e-3, 7e-3))
#plt.ylim((3e-5, 6e-5))
plt.legend()

plt.savefig('ncsx_convergence.png', dpi=300, bbox_inches='tight')
plt.close()
# import sys; sys.exit()

def movingaverage(interval, window_size=20):
    return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
plt.figure()
for i, (outdir, label) in enumerate(list(zip(outdirs, labels))[j] for j in range(3)):
    try:
        dJvals = np.load(outdir + "dJvals.npy")
        # plt.semilogy(movingaverage(dJvals[:, 0]), label=label)
        plt.semilogy(dJvals[:, 0], label=label)
    except: pass
plt.grid()
plt.legend()
plt.title("Gradient")
plt.tight_layout()
plt.savefig("ncsx_convergence_gradient.png", dpi=300)
plt.close()

import sys; sys.exit()
import seaborn as sns
import pandas as pd
sns.set_style('whitegrid')
# for i in [0, 3, 5]:
for i in range(3):
    try:
        data = np.load(outdirs[i] + "Jvals_outofsample.npy")
        # data = np.load(outdirs[i] + "Jvals_insample.npy")
        sns.kdeplot(data, label=labels[i])
    except: pass
# plt.ylim((0, 20))
plt.xlabel('$g(x^*, \\zeta)$')
plt.ylabel('pdf')
plt.title('Probability density at optimal configuration')
plt.savefig('ncsx_distribution.png', dpi=300, bbox_inches='tight')
plt.close()



# print("Configuration".ljust(20), "Mean".ljust(10), "VaR(0.9)".ljust(10), "CVaR(0.9)".ljust(10), "VaR(0.95)".ljust(10), "CVaR(0.95)".ljust(10))
print("out-of-sample")
print("Configuration".ljust(40), "Mean".ljust(22), "CVaR(0.5)".ljust(10), "CVaR(0.9)".ljust(10), "CVaR(0.95)".ljust(10), "CVaR(0.99)".ljust(10))
for i in range(0, len(labels)):
    # vals = np.load(outdirs[i] + "Jvals_insample.npy")
    vals = np.load(outdirs[i] + "Jvals_outofsample.npy")
    mean = np.mean(vals)
    std = np.std(vals)
    err = std/np.sqrt(len(vals))
    label = labels[i]
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
print("in-sample")
for i in range(0, len(labels)):
    vals = np.load(outdirs[i] + "Jvals_insample.npy")
    # vals = np.load(outdirs[i] + "Jvals_outofsample.npy")
    label = labels[i]
    var50 = np.quantile(vals, 0.5)
    cvar50 = np.mean(vals[vals>=var50])
    var90 = np.quantile(vals, 0.9)
    cvar90 = np.mean(vals[vals>=var90])
    var95 = np.quantile(vals, 0.95)
    cvar95 = np.mean(vals[vals>=var95])
    var99 = np.quantile(vals, 0.99)
    cvar99 = np.mean(vals[vals>=var99])
    print(("%s:" % label).ljust(40), ("%.3e" % np.mean(vals)).ljust(22), ("%.3e" % np.mean(cvar50)).ljust(10), ("%.3e" % np.mean(cvar90)).ljust(10), ("%.3e" % np.mean(cvar95)).ljust(10), ("%.3e" % np.mean(cvar99)).ljust(10))
