import matplotlib.pyplot as plt
import numpy as np
from pyplasmaopt import *
import os
from scipy import interpolate

def losstimes_to_lossfractions(loss_times, tmax):
    sorted_t = np.sort(loss_times)
    sorted_t = [t for t in sorted_t if t < np.inf]
    lost_percentage = [len([c for c in sorted_t if c <= t])/len(loss_times) for t in sorted_t]
    lost_percentage = [0., 0.] if len(lost_percentage) == 0 else [0.] + lost_percentage + [lost_percentage[-1]]
    avgconftime = np.mean(np.minimum(loss_times, tmax))
    return lost_percentage, [1e-5] + sorted_t + [tmax], avgconftime



config="ncsx"
# config="matt24"
E = 1000


ntcoils = 6
def filenamedet(i, IG, it=-1):
    return f"output-greene/mc13_config-ncsx_mode-deterministic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-0_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{IG}_ip-l2_optim-scipy/tracing_it_{it}_{E}eV_coilseed_{i}.npy"

def filenamestoch(i, IG, it=-1):
    return f"output-greene/mc13_config-ncsx_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-1024_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{IG}_ip-l2_optim-scipy/tracing_it_{it}_{E}eV_coilseed_{i}.npy"


exact = False
if exact:
    files = [[filenamedet(None, 0, it=0)]]
    labels = [f"NCSX Initial (Exact)"]
    files += [[filenamedet(i, IG) for i in [None]] for IG in range(8)]
    labels += [f"Deterministic (from initial guess {IG})  {s}" for s in ["Exact"] for IG in range(8)]
    files += [[filenamestoch(i, IG) for i in [None]] for IG in range(8)]
    labels += [f"Stochastic    (from initial guess {IG})  {s}" for s in ["Exact"] for IG in range(8)]
else:
    files = [[filenamedet(i, 0, it=0) for i in range(10)]]
    # labels = [f"NCSX Initial (Perturbed)"]
    labels = [f"NCSX_Initial_(Perturbed)"]
    files += [[filenamedet(i, IG) for i in range(10)] for IG in range(8)]
    # labels += [f"Deterministic (from initial guess {IG})  {s}" for s in ["Perturbed"] for IG in range(8)]
    labels += [f"Deterministic_(ig_{IG})_{s}" for s in ["Perturbed"] for IG in range(8)]
    files += [[filenamestoch(i, IG) for i in range(10)] for IG in range(8)]
    # labels += [f"Stochastic    (from initial guess {IG})  {s}" for s in ["Perturbed"] for IG in range(8)]
    labels += [f"Stochastic_(ig_{IG})_{s}" for s in ["Perturbed"] for IG in range(8)]


# print(files)

title = f"Particle losses"
subtitle = f"Proton @ {E}eV"
tmax = 1e-2
ufilter = None
# ufilter = 'pos'
# ufilter = 'neg'
if ufilter == 'pos':
    subtitle += " (all particles in fieldline direction)"
elif ufilter == 'neg':
    subtitle += " (all particles against fieldline direction)"
plt.figure(figsize=(9, 6))
plt.gca().set_axisbelow(True)
tbase = np.logspace(-5, -2, 100)
confinement_data = [tbase]
confinement_titles = ["t"]
for fils, la in zip(files, labels):
    # print(fil)
    if fils is None:
        info_all("")
        continue
    try:
        res = np.concatenate([np.load(fil) for fil in fils], axis=1)
        # print([(np.load(os.path.dirname(fil) + "/dJvals.npy")[-1, 0], np.load(os.path.dirname(fil) + "/Jvals.npy")[-1]) for fil in fils])
        import os
        # info(np.load(os.path.join(os.path.dirname(fils[0]), 'dJvals.npy'))[-1, 0])
        # for fil in fils:
        #     info(np.load(os.path.join(os.path.dirname(fil), 'dJvals.npy'))[-1, 0])
    except Exception as ex:
        # print(ex)
        continue
    if ufilter == 'pos':
        res = res[:, res[1,:]>0]
    elif ufilter == 'neg': res = res[:, res[1,:]<0]
    loss_times = res[0, :]
    us = res[1, :]
    lost_percentage, ts, avgconftime  = losstimes_to_lossfractions(loss_times, tmax)
    info_all(f"{la}".ljust(50) + f"Fraction of escaped particles: {(100*len([t for t in loss_times if t<np.inf])/len(loss_times)):.2f}%, Avg conf time: {avgconftime:.4f}")
    if 'Stochastic' in la:
        p = plt.semilogx(ts, lost_percentage, ':', drawstyle='steps-post', label=la, linewidth=2)
    elif 'NCSX' in la:
        p = plt.semilogx(ts, lost_percentage, drawstyle='steps-post', label=la, linewidth=4)
    else:
        p = plt.semilogx(ts, lost_percentage, drawstyle='steps-post', label=la, linewidth=2)
    d = p[-1].get_data() 
    confinement_data += [interpolate.interp1d(d[0], d[1])(tbase)]
    confinement_titles += [la]

np.savetxt(f"confinement_{E}.txt", np.asarray(confinement_data).T, delimiter=";", header=";".join(confinement_titles), comments="")
import sys; sys.exit()
plt.xlabel('Time')
plt.ylabel('Loss percentage')
plt.suptitle(title)
plt.title(subtitle)
plt.xlim((1e-4, tmax))
plt.ylim((0, 0.25))
plt.legend()
plt.savefig(title.replace(" ", "-").replace(".","p") + subtitle + '_exact_' + str(exact) + '.png', bbox_inches='tight', dpi=300)
# plt.show()
# plt.close()
