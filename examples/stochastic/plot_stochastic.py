from pyplasmaopt import *
from math import log10, floor, ceil
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)


# plt.figure(figsize=(4,5))

# outdirs = [
#     'output/ncsx_atopt_mode-deterministic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-0_noutsamples-1024_seed-1_sigma-0p01_length_scale-0p2/',
#     'output/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-1024_noutsamples-1024_seed-1_sigma-0p01_length_scale-0p2/',
# ]

# outdir = 'output/ncsx_atopt_mode-deterministic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-0_noutsamples-512_seed-1_sigma-0p01_length_scale-0p2/'
# Jvals = np.loadtxt(outdir + "Jvals.txt")
# out_of_sample_means = np.loadtxt(outdir + "out_of_sample_means.txt")
# plt.semilogy(Jvals, label="Det - J")
# plt.semilogy(out_of_sample_means, label="Det - OoS Mean")
# outdir = 'output/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-16_noutsamples-512_seed-1_sigma-0p01_length_scale-0p2/'
# Jvals = np.loadtxt(outdir + "Jvals.txt")
# out_of_sample_means = np.loadtxt(outdir + "out_of_sample_means.txt")
# plt.semilogy(Jvals, label="Stochastic - J")
# plt.semilogy(out_of_sample_means, label="Stochastic - OoS Mean")
# plt.show()
# import sys; sys.exit()





outdirs = [
    'output/ncsx_atopt_mode-deterministic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-0_noutsamples-512_seed-1_sigma-0p01_length_scale-0p2/',
    'output/ncsx_atopt_mode-stochastic_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-16_noutsamples-512_seed-1_sigma-0p01_length_scale-0p2/',
    'output/ncsx_atopt_mode-cvar0p95_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-1024_noutsamples-512_seed-1_sigma-0p01_length_scale-0p2/',
]

labels = ["Deterministic", "Stochastic", "CVaR"]
data_J = [np.loadtxt(outdir + "Jvals_outofsample.txt") for outdir in outdirs]
data_QS = [np.loadtxt(outdir + "QSvsBS_outofsample.txt") for outdir in outdirs]


plt.figure(figsize=(12, 5))
ax = plt.subplot(121)
bins = 100
plt.title("Objective values")
for vals, label in zip(data_J, labels):
    plt.hist(vals[1:], bins, density=True, alpha=0.75, log=False, label=label)
# plt.xscale('log')
# ax.set_xscale('log')
# ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
plt.legend()

ax = plt.subplot(122)
plt.title("$B_{\\mathrm{coils}}-B_\\mathrm{QS}$")
for vals, label in zip(data_QS, labels):
    plt.hist(vals[1:], bins, density=True, alpha=0.75, log=False, label=label)
# plt.xscale('log')
# ax.set_xscale('log')
# ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
plt.legend()

# plt.savefig(outdirs[0] + "/histogram_comparison.png", dpi=300)
# plt.savefig(outdirs[1] + "/histogram_comparison.png", dpi=300)
plt.show()



print("Configuration".ljust(20), "Mean".ljust(10), "VaR(0.9)".ljust(10), "CVaR(0.9)".ljust(10), "VaR(0.95)".ljust(10), "CVaR(0.95)".ljust(10))
for vals, label in zip(data_J, labels):
    vals = vals[1:]
    var90 = np.quantile(vals, 0.9)
    cvar90 = np.mean(list(v for v in vals if v >= var90))
    var95 = np.quantile(vals, 0.95)
    cvar95 = np.mean(list(v for v in vals if v >= var95))
    # print(("%s:" % label).ljust(20), "Mean=%.6e," % np.mean(vals), "VaR(0.9)=%.10e," % var, "CVaR(0.9)=%.10e" % cvar)
    print(("%s:" % label).ljust(20), ("%.4e" % np.mean(vals)).ljust(10), ("%.4e" % np.mean(var90)).ljust(10), ("%.4e" % np.mean(cvar90)).ljust(10), ("%.4e" % np.mean(var95)).ljust(10), ("%.4e" % np.mean(cvar95)).ljust(10))
