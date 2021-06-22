import matplotlib.pyplot as plt
import numpy as np
from pyplasmaopt import *
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.surfaceobjectives import Area, boozer_surface_residual, ToroidalFlux
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.biotsavart import BiotSavart
import sys
import shutil
import os
import logging as lg

def plot(s, filename):
    return
    from mayavi import mlab
    mlab.options.offscreen = True
    s.plot(show=False)
    mlab.view(azimuth=45, elevation=45, distance=8)
    mlab.savefig(filename, magnification=4)
    mlab.close()
    mlab.options.offscreen = False

import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--ig", type=int, default=0)
parser.add_argument("--order", type=int, default=6)
parser.add_argument("--sigma", type=str, default="0p01")
parser.add_argument("--forplotting", dest="forplotting", default=False, action="store_true")
args, _ = parser.parse_known_args()
forplotting = args.forplotting

sigma = args.sigma
def get_dir(n, ig, ntcoils):
    ppp = 20 if ntcoils < 10 else 15
    if n == 1:
        return f"output-greene/mc13_config-ncsx_mode-deterministic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-0_noutsamples-1024_seed-0_sigma-{sigma}_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/"
    else:
        return f"output-greene/mc13_config-ncsx_mode-stochastic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-{n}_noutsamples-1024_seed-0_sigma-{sigma}_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/"

outdir = get_dir(args.n, args.ig, args.order)

it = -1

info(outdir)

sys.argv = sys.argv[:1] + [str(s) for s in np.loadtxt(outdir + 'argv.txt', dtype=np.dtype('<U26'))] 
from objective_stochastic import stochastic_get_objective
obj, args = stochastic_get_objective()
savedir = outdir + "/iotas/"
shutil.rmtree(savedir, ignore_errors=True)
os.makedirs(savedir)
logger = lg.getLogger('PyPlasmaOpt')
logger.removeHandler(logger.handlers[1])

set_file_logger(savedir + 'surfacelog.txt')
obj.noutsamples = 0 if forplotting else 128
x = np.load(outdir + "xiterates.npy")[it, :]
info(f'||dJ||={np.load(outdir + "dJvals.npy")[it, 0]}')
obj.outdir = outdir
obj.update(x)
# obj.callback(x)
# obj.plot(f'coils-{it}.png')

coils = obj.stellarator.coils
currents = obj.stellarator.currents
ma = obj.ma
iota = obj.qsf.iota

G = 2. * np.pi * np.sum(np.abs(currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
bs_tf = BiotSavart(coils, currents)
bs = BiotSavart(coils, currents)


nfp = 3
stellsym = True
ntor = 5
mpol = 5
nphi = (2-stellsym)*ntor + 1
ntheta = 2*mpol + 1

phis = np.linspace(0, 1/((1+stellsym)*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1.0, ntheta, endpoint=False)


sold = None
surfaces_is = []
allres = []


s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, quadpoints_phi=phis, quadpoints_theta=thetas)
l = 0.01
s.fit_to_curve(ma, l, flip_theta=True)

tf = ToroidalFlux(s, bs_tf)
ar = Area(s)
# ar_target = ar.J()
info(f'Area={ar.J():.4f}')
ar_target = 0.6
# boozer_surface = BoozerSurface(bs, s, ar, ar_target) 

# res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=3000, constraint_weight=100., iota=iota, G=G)
# info(f"iota={res['iota']:.3f}, sqrt(tf)={np.sqrt(tf.J()):.3e}, area={ar.J():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
# res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
# info(f"iota={res['iota']:.3f}, sqrt(tf)={np.sqrt(tf.J()):.3e}, area={ar.J():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")

# print(f"iota on closest surface={res['iota']:.3e}")
from tangent_map import TangentMap

t = TangentMap(bs, ma, rtol=1e-12, atol=1e-12,
           bvp_tol=1e-8, tol=1e-12,
           verbose=0, nphi_guess=100,
           maxiter=50, method='RK45')
tiota = t.compute_iota()[0]
print(f"iota from tangent map={tiota:.3e}")

obj.compute_out_of_sample()
N = obj.noutsamples
iotas = np.zeros((N, ))
for i in range(N):
    info(f"Sample {i}")
    bs_pert = obj.stochastic_qs_objective_out_of_sample.J_BSvsQS_perturbed[i].biotsavart
    ma_points = (2*(nfp*ntor)+1) * 10
    madata_pert = find_magnetic_axis(bs_pert, ma_points, np.linalg.norm(ma.gamma()[0, 0:2]), output='cartesian')
    ma_pert = CurveRZFourier(ma_points, ma.order, 1, False)
    ma_pert.least_squares_fit(madata_pert)
    t = TangentMap(bs_pert, ma_pert, rtol=1e-12, atol=1e-12,
               bvp_tol=1e-8, tol=1e-12,
               verbose=0, nphi_guess=100,
               maxiter=50, method='RK45')
    tiota = t.compute_iota()[0]
    print(f"iota from tangent map={tiota:.3e}")
    iotas[i] = tiota
print(np.mean(iotas), np.std(iotas))
np.save(savedir + "/iotas.npy", iotas)
