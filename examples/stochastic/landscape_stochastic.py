import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker

from pyplasmaopt import *
from objective_stochastic import stochastic_get_objective
import numpy as np

import sys
sys.argv += [ 
    'optimisation_stochastic.py',
    '--ppp', '20',
    '--at-optimum',
    '--output', 'ncsx_igs',
    '--Nt_ma', '4',
    '--Nt_coils', '6',
    '--ninsamples', '4096',
    '--noutsamples', '4096',
    '--seed', '1',
    '--ip', 'l2',
    '--sigma', '0.01',
    '--length-scale', '0.2',
    '--distribution', 'gaussian',
    '--ig', '0',
    '--tikhonov', '0',
    '--opt', 'scipy',
    '--mode', 'stochastic',
    '--alen', '0.1',
    '--curvature', '1e-6',
    '--distw', '1000']

import os
os.makedirs('landscape', exist_ok=True)

xa = np.load("output-greene/ncsx_igs_atopt_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-1e-06_sobolev-0p0_alen-0p1_distw-1000p0_ig-0_ip-l2_optim-scipy/xiterates.npy")[-1, :]
xb = np.load("output-greene/ncsx_igs_atopt_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-1e-06_sobolev-0p0_alen-0p1_distw-1000p0_ig-1_ip-l2_optim-scipy/xiterates.npy")[-1, :]
xc = np.load("output-greene/ncsx_igs_atopt_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-1e-06_sobolev-0p0_alen-0p1_distw-1000p0_ig-2_ip-l2_optim-scipy/xiterates.npy")[-1, :]

info(f"|xa-xb|={np.linalg.norm(xa-xb)}")
info(f"|xb-xc|={np.linalg.norm(xb-xc)}")
info(f"|xa-xc|={np.linalg.norm(xa-xc)}")

obj, args = stochastic_get_objective()
obj.outdir = 'landscape/'
obj.update(xa)
info(f"obj(xa)={obj.res}")
if comm.rank == 0:
    obj.plot('xa.png')
obj.update(xb)
info(f"obj(xb)={obj.res}")
if comm.rank == 0:
    obj.plot('xb.png')
obj.update(xc)
if comm.rank == 0:
    obj.plot('xc.png')
info(f"obj(xc)={obj.res}")


delta = 0.5
lams = np.linspace(-delta, 1+delta, 41)
# lams = np.linspace(-delta, 1+delta, 13)
mus = lams
afaks = []
bfaks = []
cfaks = []

for lam in lams:
    for mu in mus:
        afak = 1 - lam - mu
        bfak = lam
        cfak = mu
        if afak < - delta or afak > 1 + delta:
            continue
        afaks.append(afak)
        bfaks.append(bfak)
        cfaks.append(cfak)

afaks = np.asarray(afaks)
bfaks = np.asarray(bfaks)
cfaks = np.asarray(cfaks)

res = []
dres = []
for i in range(len(afaks)):
    x = afaks[i] * xa + bfaks[i] * xb + cfaks[i] * xc
    obj.update(x)
    res.append(obj.res)
    dres.append(np.linalg.norm(obj.dres))
    info(f"afak={afaks[i]:.3f}, bfak={bfaks[i]:.3f}, cfak={cfaks[i]:.3f}, obj.res={obj.res}")


if comm.rank == 0:
    np.save('landscape/afaks.npy', afaks)
    np.save('landscape/bfaks.npy', bfaks)
    np.save('landscape/cfaks.npy', cfaks)
    np.save('landscape/res.npy', res)
    np.save('landscape/dres.npy', dres)
    np.save('landscape/xa.npy', xa)
    np.save('landscape/xb.npy', xb)
    np.save('landscape/xc.npy', xc)

    afaks = np.load('landscape/afaks.npy')
    bfaks = np.load('landscape/bfaks.npy')
    cfaks = np.load('landscape/cfaks.npy')
    res = np.load('landscape/res.npy')
    dres = np.load('landscape/dres.npy')


    plt.tricontourf(bfaks, cfaks, res)
    plt.colorbar()
    plt.savefig('landscape/res.png')
    plt.close()

    plt.tricontourf(bfaks, cfaks, np.log10(res))
    plt.colorbar()
    plt.savefig('landscape/reslog.png')
    plt.close()

    plt.tricontourf(bfaks, cfaks, dres)
    plt.colorbar()
    plt.savefig('landscape/dres.png')
    plt.close()

    plt.tricontourf(bfaks, cfaks, np.log10(dres))
    plt.colorbar()
    plt.savefig('landscape/dreslog.png')
    plt.close()
    # plt.show()


    # res = []
    # dres = []
    # lams = np.linspace(-0.5, 1.5, 81) 
    # info(f"lams={lams}")
    # for lam in lams:
    #     x = lam * xb + (1-lam) * xa
    #     obj.update(x)
    #     res.append(obj.res)
    #     dres.append(np.linalg.norm(obj.dres))
    #     info(f"lam={lam:.3f}, obj.res={obj.res}")

    # from mpi4py import MPI
    # comm = MPI.COMM_WORLD
    # import matplotlib.pyplot as plt
    # if comm.rank == 0:
    #     plt.semilogy(lams, res)
    #     plt.savefig('landscape.png')
    #     plt.close()
    #     plt.semilogy(lams, dres)
    #     plt.savefig('landscape_grad.png')
    #     plt.close()

    print(res)
    print(dres)
    # import IPython; IPython.embed()
