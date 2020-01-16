from pyplasmaopt import *
from math import log10, floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)

from problem2_objective import get_objective
obj, args = get_objective()
outdir = obj.outdir
# outdir = "output-deterministic-sigma-0.00300/"
# outdir = "output-stochastic-sigma-0.00300/"
# outdir = "output-stochastic-sigma-0.00300-seed-2-8/"

Jvals = np.loadtxt(outdir + "Jvals.txt")
dJvals = np.loadtxt(outdir + "dJvals.txt")
Jvals_quantiles = np.loadtxt(outdir + "Jvals_quantiles.txt")
Jvals_no_noise = np.loadtxt(outdir + "Jvals_no_noise.txt")
L2s = np.loadtxt(outdir + "L2s.txt")
H1s = np.loadtxt(outdir + "H1s.txt")
xiterates = np.loadtxt(outdir + "xiterates.txt")
niterates = len(Jvals)

plt.figure()
plt.semilogy(Jvals, label="J")
plt.semilogy(Jvals_quantiles[:, 0], label=r"$10\%$ Quantile")
plt.semilogy(Jvals_quantiles[:, 1], label="Mean")
plt.semilogy(Jvals_quantiles[:, 2], label=r"$90\%$ Quantile")
plt.semilogy(Jvals_no_noise,  label="No noise")
plt.legend()
plt.grid()
ymin, ymax = plt.ylim()
logmin = floor(log10(ymin))
logmax = ceil(log10(ymax))
ax = plt.gca()
ax.set_yticks([10**(i) for i in range(logmin, logmax+1)])
plt.title("Convergence")
plt.tight_layout()
plt.savefig(outdir + "convergence_objective.png", dpi=300)
plt.close()


plt.figure()
plt.semilogy(dJvals[:, 0], label=r"$dJ$")
plt.semilogy(dJvals[:, 1], label=r"$dJ_{etabar}$")
plt.semilogy(dJvals[:, 2], label=r"$dJ_{ma}$")
plt.semilogy(dJvals[:, 3], label=r"$dJ_{current}$")
plt.semilogy(dJvals[:, 4], label=r"$dJ_{coil}$")
plt.grid()
plt.legend()
plt.title("Convergence")
plt.tight_layout()
plt.savefig(outdir + "convergence_gradient.png", dpi=300)
plt.close()


locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1) 
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,1.0, ))


plt.figure(figsize=(10, 5))
bins = 100
ax = plt.subplot(121)
plt.title("$\\frac{1}{2}||B_{BS}-B_{QS}||^2$")
plt.hist(L2s[1:], bins, density=True, facecolor='g', alpha=0.75, log=True)
plt.xscale('log')
ymax = plt.ylim()[1]
plt.vlines(L2s[0], 0, ymax, color='r', label="Without pertubation")
plt.vlines(np.mean(L2s[1:]), 0, ymax, color='y', label="Mean")
xmin, xmax = plt.xlim()
logmin = floor(log10(xmin))
logmax = ceil(log10(xmax))
plt.xlim((10**logmin, 10**logmax))
import matplotlib.ticker as ticker
ax.set_xscale('log')
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.))
# ax.set_xticks([10**(i) for i in range(logmin, logmax+1)])
# plt.xlim((0.5 * min(L2s), 2 * max(L2s)))
plt.legend()

ax = plt.subplot(122)
plt.title("$\\frac{1}{2}||\\nabla B_{BS}-\\nabla B_{QS}||^2$")
plt.hist(H1s[1:], bins, density=True, facecolor='b', alpha=0.75, log=True)
plt.xscale('log')
ymax = plt.ylim()[1]
plt.vlines(H1s[0], 0, ymax, color='r', label="Without pertubation")
plt.vlines(np.mean(H1s[1:]), 0, ymax, color='y', label="Mean")
xmin, xmax = plt.xlim()
xogmin = floor(log10(xmin))
logmax = ceil(log10(xmax))
plt.xlim((10**logmin, 10**logmax))
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
# ax.set_xticks([10**(i) for i in range(logmin, logmax+1)])
plt.legend()

# plt.suptitle("$\\sigma=%.4f$" % sigma_perturb)
plt.savefig(outdir + "histogram.png", dpi=300)
plt.close()
max_curvatures = []
mean_curvatures = []
max_torsions = []
mean_torsions = []
min_distances = []
min_distances_points = []
# import IPython; IPython.embed()
stellarator = obj.stellarator
for i in range(niterates):
    obj.set_dofs(xiterates[i, :])
    max_curvatures.append(max(np.max(c.kappa) for c in stellarator._base_coils))
    mean_curvatures.append(np.mean([np.mean(c.kappa) for c in stellarator._base_coils]))
    max_torsions.append(max(np.max(np.abs(c.torsion)) for c in stellarator._base_coils))
    mean_torsions.append(np.mean([np.mean(np.abs(c.torsion)) for c in stellarator._base_coils]))
    num_total_coils = len(stellarator.coils)
    distances = np.zeros((num_total_coils, num_total_coils))
    argmin0 = np.zeros((num_total_coils, num_total_coils), dtype=np.int64)
    argmin1 = np.zeros((num_total_coils, num_total_coils), dtype=np.int64)
    distances[:, :] = 1e10
    from scipy.spatial.distance import cdist
    for i in range(num_total_coils):
        for j in range(i-1):
            dists = cdist(stellarator.coils[i].gamma, stellarator.coils[j].gamma)
            np.fill_diagonal(dists, 1e10)
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            distances[i, j] = np.min(dists)
            argmin0[i, j] = idx[0]
            argmin1[i, j] = idx[1]
    idx = np.unravel_index(np.argmin(distances), distances.shape)
    min_distances.append(np.min(distances))
    min_distances_points.append((stellarator.coils[idx[0]].gamma[argmin0[idx[0], idx[1]], :], stellarator.coils[idx[1]].gamma[argmin1[idx[0], idx[1]], :]))


plt.figure()
plt.semilogy(mean_curvatures, label="Mean curvature")
plt.semilogy(mean_torsions, label="Mean torsion")
plt.semilogy(max_curvatures, label="Max curvature")
plt.semilogy(max_torsions, label="Max torsion")
plt.ylim((1, plt.ylim()[1]))
plt.grid()
plt.legend()
plt.savefig(outdir + "curvature_torsion.png", dpi=300)
plt.close()

plt.figure()
plt.plot(min_distances, label="Minimum distance")
plt.ylim(0, 0.15)
plt.grid()
plt.legend()
plt.savefig(outdir + "distance.png", dpi=300)
plt.close()
# import IPython; IPython.embed()


# ax = None
# for i in range(0, len(obj.stellarator.coils)):
#     ax = obj.stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(obj.stellarator._base_coils)])
# obj.ma.plot(ax=ax, show=False, closed_loop=False)
# x = min_distances_points[-1][0]
# y = min_distances_points[-1][1]
# plt.plot([x[0], y[0]],[x[1], y[1]],[x[2], y[2]])
# ax.view_init(elev=90., azim=0)
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-1, 1)
# plt.show()
# plt.savefig(obj.outdir + filename, dpi=300)
# plt.close()
