from pyplasmaopt import *
from math import log10, floor, ceil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
outdirs = [
    "output-4th-power_at_optimum-False_mode-deterministic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-100_curvature_penalty-0p0_torsion_penalty-1e-07_tikhonov-1e-05_sobolev-1e-07_arclength-0p0_minimum_distance-0p1_distance_weight-10p0",
    "output-4th-power_at_optimum-False_mode-stochastic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-100_curvature_penalty-0p0_torsion_penalty-1e-07_tikhonov-1e-05_sobolev-1e-07_arclength-0p0_minimum_distance-0p1_distance_weight-10p0"
]
# outdirs = [
#     "output-4th-power_at_optimum-False_mode-deterministic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-100_curvature_penalty-0p0_torsion_penalty-1e-07_tikhonov-1e-05_sobolev-0p0_arclength-0p0_minimum_distance-0p1_distance_weight-10p0",
#     "output-4th-power_at_optimum-False_mode-stochastic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-100_curvature_penalty-0p0_torsion_penalty-1e-07_tikhonov-1e-05_sobolev-0p0_arclength-0p0_minimum_distance-0p1_distance_weight-10p0"
# ]

outdirs = [
    "output-4th-power_at_optimum-False_mode-stochastic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-100_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p01_sobolev-0p0_arclength-0p0_minimum_distance-0p1_distance_weight-0p0",
    "output-4th-power_at_optimum-False_mode-deterministic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-100_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p01_sobolev-0p0_arclength-0p0_minimum_distance-0p1_distance_weight-0p0",
]
outdirs = [
    "output_at_optimum-False_mode-stochastic_sigma-0p01_length_scale-0p2_seed-1_ppp-10_nsamples-100_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p0_sobolev-0p0_arclength-0p0_minimum_distance-0p04_distance_weight-0p0",
    "output_at_optimum-False_mode-cvar0p9_sigma-0p01_length_scale-0p2_seed-1_ppp-10_nsamples-100_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p0_sobolev-0p0_arclength-0p0_minimum_distance-0p04_distance_weight-0p0",
]

outdirs = [
    "output_at_optimum-False_mode-stochastic_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-200_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p0_sobolev-0p0_arclength-0p0_minimum_distance-0p1_distance_weight-0p0",
    "output_at_optimum-False_mode-cvar0p9_sigma-0p003_length_scale-0p2_seed-3_ppp-20_nsamples-200_curvature_penalty-0p0_torsion_penalty-0p0_tikhonov-0p0_sobolev-0p0_arclength-0p0_minimum_distance-0p1_distance_weight-0p0",
]

labels = ["stochastic", "CVaR"]
data = [np.loadtxt(outdir + "/Jvals_perturbed_more.txt") for outdir in outdirs]
data_h1 = [np.loadtxt(outdir + "/H1s.txt") for outdir in outdirs]
data_l2 = [np.loadtxt(outdir + "/L2s.txt") for outdir in outdirs]
data = [np.loadtxt(outdir + "/Jvals_perturbed.txt")[-1,:] for outdir in outdirs]

plt.figure(figsize=(12, 5))
ax = plt.subplot(121)
bins = 100
plt.title("Objective values")
for vals, label in zip(data, labels):
    plt.hist(vals[1:], bins, density=True, alpha=0.75, log=True, label=label)
plt.xscale('log')
ax.set_xscale('log')
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
plt.legend()

ax = plt.subplot(122)
plt.title("$\\frac{1}{2}||\\nabla B_{BS}-\\nabla B_{QS}||^2$")
for vals, label in zip(data_h1, labels):
    plt.hist(vals[1:], bins, density=True, alpha=0.75, log=False, label=label)
# plt.xscale('log')
ax.set_xscale('log')
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
plt.legend()

plt.savefig(outdirs[0] + "/histogram_comparison.png", dpi=300)
plt.savefig(outdirs[1] + "/histogram_comparison.png", dpi=300)
# plt.show()


for vals, label in zip(data, labels):
    vals = vals[1:]
    var = np.quantile(vals, 0.9)
    cvar = np.mean(list(v for v in vals if v >= var))
    print(("%s:" % label).ljust(20), "Mean=%.6e," % np.mean(vals), "VaR(0.9)=%.10e," % var, "CVaR(0.9)=%.10e" % cvar)
import IPython; IPython.embed()
