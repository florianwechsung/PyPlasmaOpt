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

import os
os.makedirs('mcconvergence', exist_ok=True)

def get_dir(n, ig, ntcoils):
    ppp = 20 if ntcoils < 10 else 15
    if n == 1:
        return f"output-greene/mc13_config-ncsx_mode-deterministic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-0_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/"
    else:
        return f"output-greene/mc13_config-ncsx_mode-stochastic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-{n}_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/"

ns = np.asarray([4**i for i in range(0, 6)])
igs = np.asarray(list(range(8)))



# from shutil import copyfile
# for i in range(0, N):
#     copyfile(outdirs[i] + "optim.png", f"mcconvergence/optim_{ns[i]}.png")

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,5))
# fig.suptitle('In-sample vs out-of-sample convergence')
# for ntcoils in [6]:
#     color1 = next(ax1._get_lines.prop_cycler)['color']
#     color2 = next(ax1._get_lines.prop_cycler)['color']
#     for ig in igs[:8]:
#         means_insample = []
#         means_outsample = []
#         ns = [1, 4, 16, 64, 256, 1024]
#         for n in ns:
#             outdir = get_dir(n, ig, ntcoils)
#             if n>1:
#                 vals = np.load(outdir + "Jvals_insample.npy")
#                 means_insample.append(np.mean(vals))
#             else:
#                 means_insample.append(np.load(outdir + "Jvals.npy")[-1])
#             vals = np.load(outdir + "Jvals_outofsample.npy")
#             means_outsample.append(np.mean(vals))
#         means_insample = np.asarray(means_insample)
#         means_outsample = np.asarray(means_outsample)
#         errs = np.abs(means_insample-means_outsample)
#         C = np.mean(errs*np.sqrt(ns))
#         ax1.semilogx(ns, means_insample, color=color1)
#         ax1.semilogx(ns, means_outsample, ':', color=color2)
#         ax2.loglog(ns, errs)
# ax1.set_xlabel('Sample size')
# ax1.set_ylabel('Sample mean')
# ax1.legend()
# ax2.loglog(ns, C/np.sqrt(ns), ":", label=r'$\frac{1}{\sqrt{N_{MC}}}$')
# ax2.set_xlabel('Sample size')
# ax2.set_ylabel(r'In-sample vs Out-of-sample difference')
# ax2.legend()
# plt.savefig(f"mcconvergence/sampleconvergence.png", dpi=600, bbox_inches='tight')
# plt.close()

# plt.show()


import seaborn as sns
import pandas as pd
plt.figure(figsize=(5,5))
sns.set_style('whitegrid')

labels_density = []
data_density = []
ax = plt.gca()
for n in [1, 4, 64, 1024]:
    for ntcoils in [6]:
        color = next(ax._get_lines.prop_cycler)['color']
        for ig in igs[:8]:
            # data = np.load(get_dir(n, ig, ntcoils) + "QSvsBS_outofsample.npy")
            data = np.load(get_dir(n, ig, ntcoils) + "Jvals_outofsample.npy")
            print(f"n={n:4d}, mean={np.mean(data)}")
            if ig == 0:
                p = sns.kdeplot(data, label=f"N = {n}, CoilOrder = {ntcoils}", color=color)
            else:
                p = sns.kdeplot(data, color=color)
            d = p.get_lines()[-1].get_data() 
            data_density += d
            labels_density += [f"x_n_{n}_order_{ntcoils}_ig_{ig}", f"y_n_{n}_order_{ntcoils}_ig_{ig}"]

plt.xlabel('Objective value')
plt.ylabel('pdf')
plt.legend()
plt.title('Probability density at optimal configuration')
plt.xlim((0, 0.04))
plt.savefig(f"mcconvergence/density.pdf", dpi=600, bbox_inches='tight')
np.savetxt("mcconvergence/density.txt", np.asarray(data_density).T, delimiter=";", header=";".join(labels_density), comments="")
plt.show()


# data_convergence = []
# plt.figure(figsize=(5,5))
# ax = plt.gca()
# for n in [1, 4, 64, 1024]:
#     for ntcoils in [6]:
#         color = next(ax._get_lines.prop_cycler)['color']
#         for ig in igs[:8]:
#             colors = next(ax._get_lines.prop_cycler)['color']
#             outdir = get_dir(n, ig, ntcoils)
#             Jvals = np.load(outdir + "Jvals.npy")
#             out_of_sample_means = np.load(outdir + "out_of_sample_means.npy")

#             data_convergence.append(Jvals)
#             data_convergence.append(out_of_sample_means)

#             plt.semilogy(Jvals, "--", color=colors, linewidth=2)
#             plt.semilogy(out_of_sample_means, "-", color=colors, linewidth=1.5)

# # tmp = np.zeros((max(len(l) for l in data_convergence), 2*(len(outdirs))))
# # tmp[:, :] = np.nan
# # for i, _ in enumerate(outdirs):
# #     tmp[:len(data_convergence[2*i]), 2*i] = data_convergence[2*i]
# # labels = ";".join(sum([[f"is{n}", f"oos{n}"] for n in ns], []))
# # np.savetxt("mcconvergence/convergence.txt", tmp, delimiter=";", header=labels, comments="")
# # import IPython; IPython.embed()
# # import sys; sys.exit()
# plt.grid(which='both')
# plt.xlabel('Iteration')
# # plt.title('Objective value and Out-of-Sample mean')
# plt.ylim((1e-3, 7e-3))
# #plt.ylim((3e-5, 6e-5))
# plt.legend()

# plt.savefig('ncsx_convergence.png', dpi=600, bbox_inches='tight')
# plt.show()
# plt.close()
# import IPython; IPython.embed()
