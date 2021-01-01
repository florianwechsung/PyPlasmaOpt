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
    "output-greene/ncsx_atopt_mode-deterministic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-0_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    "output-greene/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-16_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    "output-greene/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-64_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    "output-greene/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-256_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    "output-greene/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-1024_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    "output-greene/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    "output-greene/ncsx-from1024_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    # "output-greene/ncsx_atopt_mode-cvar0p5_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    # "output-greene/ncsx_atopt_mode-cvar0p9_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    # "output-greene/ncsx_atopt_mode-cvar0p95_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    "output-greene/ncsx-from1024_atopt_mode-cvar0p5_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    "output-greene/ncsx-from1024_atopt_mode-cvar0p9_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
    "output-greene/ncsx-from1024_atopt_mode-cvar0p95_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2/",
]

labels = ["Deterministic",
 "Stochastic(n=16)",
 "Stochastic(n=64)",
 "Stochastic(n=256)",
 "Stochastic(n=1024)",
 "Stochastic(n=4096)",
 "Stochastic(n=4096 ig 1024)",
 # "CVaR(0.50)",
 # "CVaR(0.90)",
 # "CVaR(0.95)",
 "CVaR(0.50 ig 1024)",
 "CVaR(0.90 ig 1024)",
 "CVaR(0.95 ig 1024)"]


outdirs = [
    "output-greene/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-1024_noutsamples-1024_seed-1_sigma-0p03_length_scale-0p2/",
    "output-greene/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p03_length_scale-0p2/",
    "output-greene/ncsx_atopt_mode-cvar0p5_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p03_length_scale-0p2/",
    "output-greene/ncsx_atopt_mode-cvar0p9_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p03_length_scale-0p2/",
    "output-greene/ncsx_atopt_mode-cvar0p95_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p03_length_scale-0p2/",
]

labels = [
 "Stochastic(n=1024)",
 "Stochastic(n=4096)",
 "CVaR(0.50)",
 "CVaR(0.90)",
 "CVaR(0.95)",
]
outdirs = [
    "output/temp_atopt_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-32_noutsamples-32_seed-1_sigma-0p003_length_scale-0p2/",
    "output/temp_atopt_mode-stochastic_distribution-uniform_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-32_noutsamples-32_seed-1_sigma-0p003_length_scale-0p2/",
    "output/temp_atopt_mode-cvar0p9_distribution-uniform_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-32_noutsamples-32_seed-1_sigma-0p003_length_scale-0p2/",
]

labels = [
 "Stochastic(Gaussian)",
 "Stochastic(Uniform)",
 "CVaR 0.9 (Uniform)",
]

outdirs = [
    "output/temp_atopt_mode-stochastic_distribution-uniform_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-64_noutsamples-64_seed-1_sigma-0p003_length_scale-0p2/",
    "output/temp_atopt_mode-cvar0p5_distribution-uniform_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-64_noutsamples-64_seed-1_sigma-0p003_length_scale-0p2/",
]

labels = [
 "Stochastic(Uniform)",
 "CVaR 0.5 (Uniform)",
]

outdirs = [
    "output/temp_atopt_mode-stochastic_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-2048_noutsamples-2048_seed-1_sigma-0p01_length_scale-0p2/",
    "output/temp_atopt_mode-cvar0p5_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-2048_noutsamples-2048_seed-1_sigma-0p01_length_scale-0p2/",
    "output/temp_atopt_mode-cvar0p9_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-2048_noutsamples-2048_seed-1_sigma-0p01_length_scale-0p2/",
    "output/temp_atopt_mode-cvar0p95_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-2048_noutsamples-2048_seed-1_sigma-0p01_length_scale-0p2/",
]

labels = [
 "Stochastic (Uniform)",
 "CVaR 0.50  (Uniform)",
 "CVaR 0.90  (Uniform)",
 "CVaR 0.95  (Uniform)",
]


# plt.figure(figsize=(9,5))
# ax = plt.gca()
# for i, (outdir, label) in enumerate(zip(outdirs, labels)):
#     if not i in [0, 1, 5]:
#         continue
#     Jvals = np.load(outdir + "Jvals.npy")
#     out_of_sample_means = np.load(outdir + "out_of_sample_means.npy")
#     colors = next(ax._get_lines.prop_cycler)['color']
#     plt.loglog(Jvals, "--", label=label + " - J", color=colors, linewidth=2)
#     plt.loglog(out_of_sample_means, "-", label=label + " - OoS Mean", color=colors, linewidth=1.5)
# plt.grid()
# plt.xlabel('Iteration')
# plt.title('Objective value and Out-of-Sample mean')
# plt.legend()

# plt.savefig('ncsx_convergence.png', dpi=300, bbox_inches='tight')
# plt.close()

# def movingaverage(interval, window_size=20):
#     window = np.ones(int(window_size))/float(window_size)
#     return np.convolve(interval, window, 'same')
# plt.figure()
# for i, (outdir, label) in enumerate(zip(outdirs, labels)):
#     dJvals = np.load(outdir + "dJvals.npy")
#     plt.semilogy(movingaverage(dJvals[:, 0]), label=label)
#     if i==8:
#         break
# plt.grid()
# plt.legend()
# plt.title("Gradient")
# plt.tight_layout()
# plt.savefig("ncsx_convergence_gradient.png", dpi=300)
# plt.close()


import seaborn as sns
import pandas as pd
sns.set_style('whitegrid')
for i in [0, 1, 2]:
    data = np.log(np.load(outdirs[i] + "Jvals_outofsample.npy"))
    data = np.load(outdirs[i] + "Jvals_outofsample.npy")
    # data = np.load(outdirs[i] + "Jvals_insample.npy")
    sns.kdeplot(data, label=labels[i])
plt.xlabel('$g(x^*, \\zeta)$')
plt.ylabel('pdf')
plt.title('Probability density at optimal configuration')
plt.savefig('ncsx_distribution.png', dpi=300, bbox_inches='tight')
plt.close()



# print("Configuration".ljust(20), "Mean".ljust(10), "VaR(0.9)".ljust(10), "CVaR(0.9)".ljust(10), "VaR(0.95)".ljust(10), "CVaR(0.95)".ljust(10))
print("out-of-sample")
print("Configuration".ljust(30), "Mean".ljust(10), "CVaR(0.5)".ljust(10), "CVaR(0.9)".ljust(10), "CVaR(0.95)".ljust(10), "CVaR(0.99)".ljust(10))
for i in range(0, len(labels)):
    # vals = np.load(outdirs[i] + "Jvals_insample.npy")
    vals = np.load(outdirs[i] + "Jvals_outofsample.npy")
    label = labels[i]
    var50 = np.quantile(vals, 0.5)
    cvar50 = np.mean(vals[vals>=var50])
    var90 = np.quantile(vals, 0.9)
    cvar90 = np.mean(vals[vals>=var90])
    var95 = np.quantile(vals, 0.95)
    cvar95 = np.mean(vals[vals>=var95])
    var99 = np.quantile(vals, 0.99)
    cvar99 = np.mean(vals[vals>=var99])
    print(("%s:" % label).ljust(30), ("%.3e" % np.mean(vals)).ljust(10), ("%.3e" % np.mean(cvar50)).ljust(10), ("%.3e" % np.mean(cvar90)).ljust(10), ("%.3e" % np.mean(cvar95)).ljust(10), ("%.3e" % np.mean(cvar99)).ljust(10))
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
    print(("%s:" % label).ljust(30), ("%.3e" % np.mean(vals)).ljust(10), ("%.3e" % np.mean(cvar50)).ljust(10), ("%.3e" % np.mean(cvar90)).ljust(10), ("%.3e" % np.mean(cvar95)).ljust(10), ("%.3e" % np.mean(cvar99)).ljust(10))
