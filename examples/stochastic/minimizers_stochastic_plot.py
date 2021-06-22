import matplotlib.pyplot as plt
import numpy as np
from pyplasmaopt import *
from mayavi import mlab
def get_dir(n, ig, ntcoils):
    ppp = 20 if ntcoils < 10 else 15
    if n == 1:
        return f"output-greene/mc13_config-ncsx_mode-deterministic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-0_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/"
    else:
        return f"output-greene/mc13_config-ncsx_mode-stochastic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-{n}_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/"



colors  = [(a/255, b/255, c/255) for (a, b, c) in [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207)]]

mlab.options.offscreen = True
mlab.figure(bgcolor=(1, 1, 1))
n = 1
# n = 4
# n = 1024
order = 6
for ig in range(8):
    try:
        outdir = get_dir(n, ig, order)
        it = -1
        import sys
        sys.argv = sys.argv[:1] + [str(s) for s in np.loadtxt(outdir + 'argv.txt', dtype=np.dtype('<U26'))] 
        from objective_stochastic import stochastic_get_objective
        obj, args = stochastic_get_objective()
        obj.noutsamples = 20
        xiterates = np.load(outdir + "xiterates.npy")
        x = xiterates[it, :]
        print(outdir)
        print(f'||dJ[{xiterates.shape[0]}]||={np.load(outdir + "dJvals.npy")[it, 0]}')
        obj.outdir = outdir
        obj.update(x)
        coils = obj.stellarator.coils
        currents = obj.stellarator.currents
        for c in obj.stellarator._base_coils:
            c.plot_mayavi(show=False, color=colors[ig])
    except:
        print(f'Skip IG {ig}')
        pass


mlab.view(azimuth=45, elevation=45)
mlab.savefig(f"ncsx_min_{n}.png", magnification=4)
# mlab.show()
