
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(edgeitems=30, linewidth=200, formatter=dict(float=lambda x: "%+.4e" % x))

# d = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}
import os
os.makedirs('surfaces', exist_ok=True)

def get_dir(n, ig, ntcoils):
    ppp = 20 if ntcoils < 10 else 15
    if n == 1:
        return f"output-greene/mc13_config-ncsx_mode-deterministic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-0_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/"
    else:
        return f"output-greene/mc13_config-ncsx_mode-stochastic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-{n}_noutsamples-1024_seed-0_sigma-0p01_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/"

n = 1024
order = 6
ig = np.argmin([np.mean(np.load(get_dir(n, ig, order) + "Jvals_outofsample.npy")) for ig in range(8)])
outdir = get_dir(n, ig, order)

it = -1
print(outdir)

import sys
sys.argv = sys.argv[:1] + [str(s) for s in np.loadtxt(outdir + 'argv.txt', dtype=np.dtype('<U26'))] 
from objective_stochastic import stochastic_get_objective
obj, args = stochastic_get_objective()
obj.noutsamples = 1
x = np.load(outdir + "xiterates.npy")[it, :]
print(f'||dJ||={np.load(outdir + "dJvals.npy")[it, 0]}')
obj.outdir = outdir
obj.update(x)

coils = obj.stellarator.coils
currents = obj.stellarator.currents
ma = obj.ma
iota = obj.qsf.iota

from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
mpol = 5
ntor = 5
nfp = 3
phis_full = np.linspace(0, 1, 5*(2*(nfp*ntor)+2), endpoint=True)
thetas_full = np.linspace(0, 1, 5*(2*(mpol)+2), endpoint=True)
sfull = SurfaceXYZTensorFourier(mpol=mpol, ntor=nfp*ntor, nfp=1, stellsym=False, quadpoints_phi=phis_full, quadpoints_theta=thetas_full)
phis_partial = np.linspace(0, 1/(2*nfp), 5*(ntor+2), endpoint=True)
thetas_partial = np.linspace(0, 1, 5*(2*(mpol)+2), endpoint=True)
spartial = SurfaceXYZTensorFourier(mpol=mpol, ntor=nfp*ntor, nfp=1, stellsym=False, quadpoints_phi=phis_partial, quadpoints_theta=thetas_partial)

sdofs = np.load(outdir + "surfacedofs.npy")

def magnetic_field_on_surface(s):
    bs = obj.biotsavart
    x = s.gamma()
    B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
    mod_B = np.linalg.norm(B, axis=2)
    return mod_B


import mayavi.mlab as mlab
mlab.options.offscreen = True

violet = (87/255, 6/255, 140/255)
copper =  (72.2/100, 45.1/100, 20/100)
for i in [1, 4, 7]:
    mlab.figure(bgcolor=(1, 1, 1))
    for c in obj.stellarator._base_coils:
        c.plot_mayavi(show=False, color=copper)
    sfull.set_dofs(sdofs[i, :])
    # sfull.plot(scalars=magnetic_field_on_surface(sfull), wireframe=False)
