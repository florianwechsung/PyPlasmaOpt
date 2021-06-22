import matplotlib.pyplot as plt
import numpy as np
from pyplasmaopt import *



for n in [1, 4, 16, 64, 256, 1024, 4096]:
    outdir = f"output-greene/mc7_config-ncsx_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-5_Nt_coils-5_ninsamples-{n}_noutsamples-4096_seed-1_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-1e-06_clen-1p0_distw-1000p0_ig-0_ip-l2_optim-scipy/"
    outdir = f"output-greene/mc7_config-ncsx_mode-stochastic_distribution-gaussian_ppp-20_Nt_ma-4_Nt_coils-6_ninsamples-{n}_noutsamples-4096_seed-1_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-1e-06_clen-1p0_distw-1000p0_ig-0_ip-l2_optim-scipy/"
    it = -1

    import sys
    sys.argv = sys.argv[:1] + [str(s) for s in np.loadtxt(outdir + 'argv.txt', dtype=np.dtype('<U26'))] 
    from objective_stochastic import stochastic_get_objective
    obj, args = stochastic_get_objective()
    x = np.load(outdir + "xiterates.npy")[it, :]
    normdJ = np.load(outdir + "dJvals.npy")[it, 0]
    print(f'normdJ on greene = {normdJ}')
    obj.outdir = outdir
    obj.update(x)
    obj.callback(x)
    obj.plot(f'coils-{it}.png')

# for IG in [0, 1, 2, 3]:
#     outdir = f"output-greene/conf_config-ncsx_mode-stochastic_distribution-gaussian_ppp-24_Nt_ma-5_Nt_coils-5_ninsamples-1024_noutsamples-1024_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-1e-06_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{IG}_ip-l2_optim-scipy/"
#     # outdir = f"output-greene/conf_config-ncsx_mode-deterministic_distribution-gaussian_ppp-24_Nt_ma-5_Nt_coils-5_ninsamples-0_noutsamples-0_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-1e-06_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{IG}_ip-l2_optim-scipy/"
#     it = -1

#     import sys
#     sys.argv = sys.argv[:1] + [str(s) for s in np.loadtxt(outdir + 'argv.txt', dtype=np.dtype('<U26'))] 
#     from objective_stochastic import stochastic_get_objective
#     obj, args = stochastic_get_objective()
#     x = np.load(outdir + "xiterates.npy")[it, :]
#     obj.outdir = outdir
#     obj.update(x)
#     obj.callback(x)
#     obj.plot(f'coils-{it}.png')
