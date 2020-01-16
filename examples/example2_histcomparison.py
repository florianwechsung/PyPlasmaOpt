from pyplasmaopt import *
from math import log10, floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)

outdirs = [
    "output-4th-power_at_optimum-False_mode-stochastic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-200_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p001",
    "output-4th-power_at_optimum-False_mode-deterministic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-200_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p001"
]
outdirs = [
    "output-4th-power_at_optimum-False_mode-stochastic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-200_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p0",
    "output-4th-power_at_optimum-False_mode-deterministic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-200_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p0"
]

labels = ["stochastic", "deterministic"]
data = [np.loadtxt(outdir + "/Jvals_perturbed_more.txt") for outdir in outdirs]
#data = [np.loadtxt(outdir + "/Jvals_perturbed.txt") for outdir in outdirs]

plt.figure(figsize=(10, 5))
bins = 100
plt.title("Objective values")
for vals, label in zip(data, labels):
    #plt.hist(vals[-1, 1:], bins, density=True, alpha=0.75, log=True, label=label)
    plt.hist(vals[1:], bins, density=True, alpha=0.75, log=True, label=label)
plt.xscale('log')
# ymax = plt.ylim()[1]
# plt.vlines(np.mean(L2s[1:]), 0, ymax, color='y', label="Mean")
# xmin, xmax = plt.xlim()
# logmin = floor(log10(xmin))
# logmax = ceil(log10(xmax))
# plt.xlim((10**logmin, 10**logmax))
import matplotlib.ticker as ticker
ax = plt.gca()
ax.set_xscale('log')
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.))
# ax.set_xticks([10**(i) for i in range(logmin, logmax+1)])
# plt.xlim((0.5 * min(L2s), 2 * max(L2s)))
plt.legend()
plt.savefig("histogram_comparison.png", dpi=300)
plt.show()
