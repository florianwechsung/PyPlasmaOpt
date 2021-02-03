import matplotlib.pyplot as plt
import numpy as np
from pyplasmaopt import *

def losstimes_to_lossfractions(loss_times, tmax):
    sorted_t = np.sort(loss_times)
    sorted_t = [t for t in sorted_t if t < np.inf]
    lost_percentage = [len([c for c in sorted_t if c <= t])/len(loss_times) for t in sorted_t]
    lost_percentage = [0.] if len(lost_percentage) == 0 else lost_percentage + [lost_percentage[-1]]
    return lost_percentage, sorted_t + [tmax]


outdir = "output/Hk_atopt_mode-deterministic_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-1_noutsamples-8_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-0p0_ig-0_ip-l2_optim-scipy/"
# outdir = "output/Hk_atopt_mode-stochastic_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-64_noutsamples-64_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-0p0_ig-0_ip-l2_optim-scipy/"
# outdir = "output/Hk_atopt_mode-stochastic_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-1024_noutsamples-1024_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-0p0_ig-0_ip-l2_optim-scipy/"

files = ["coilseed_None.npy", "coilseed_0.npy", "coilseed_1.npy", "coilseed_2.npy", "coilseed_3.npy", "coilseed_4.npy"]
files = [outdir + fil for fil in files]
labels = ['No noise', 'Perturbed Coils 1', 'Perturbed Coils 2', 'Perturbed Coils 3', 'Perturbed Coils 4', 'Perturbed Coils 5']

files = [f"surf_coilseed_{i}.npy" for i in [None] + list(range(5))]
labels = ['No noise', 'Perturbed Coils 1', 'Perturbed Coils 2', 'Perturbed Coils 3', 'Perturbed Coils 4', 'Perturbed Coils 5']

title = "Particle losses"
tmax = 1e-2
plt.figure()
for fil, la in zip(files, labels):
    try:
        res = np.load(fil)
    except:
        continue
    loss_times = res[0, :]
    us = res[1, :]
    info_all(f"{la}, Fraction of escaped particles: {(100*len([t for t in loss_times if t<np.inf])/len(loss_times)):.2f}%")
    lost_percentage, ts  = losstimes_to_lossfractions(loss_times, tmax)
    plt.semilogx(ts, lost_percentage, drawstyle='steps-post', label=la)
plt.xlabel('Time')
plt.ylabel('Loss percentage')
plt.title(title)
plt.xlim((1e-5, tmax))
plt.ylim((0, 0.5))
plt.legend()
plt.show()
# plt.savefig(outdir + filename + title.replace(' ', '-') + '.png')
# plt.close()
