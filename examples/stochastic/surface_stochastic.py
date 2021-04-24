import matplotlib.pyplot as plt
import numpy as np
from pyplasmaopt import *
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.surfaceobjectives import Area, boozer_surface_residual, ToroidalFlux
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.biotsavart import BiotSavart

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

print(outdir)

import sys
sys.argv = sys.argv[:1] + [str(s) for s in np.loadtxt(outdir + 'argv.txt', dtype=np.dtype('<U26'))] 
from objective_stochastic import stochastic_get_objective
obj, args = stochastic_get_objective()
obj.noutsamples = 0 if forplotting else 20
x = np.load(outdir + "xiterates.npy")[it, :]
print(f'||dJ||={np.load(outdir + "dJvals.npy")[it, 0]}')
obj.outdir = outdir
obj.update(x)
savedir = outdir + "/surfaces2/"
import shutil
shutil.rmtree(savedir, ignore_errors=True)
import os
os.makedirs(savedir)
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
ntor = 10
mpol = 5
nphi = (2-stellsym)*ntor + 1
ntheta = 2*mpol + 1

phis = np.linspace(0, 1/((1+stellsym)*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1.0, ntheta, endpoint=False)


sold = None
surfaces_is = []
allres = []

nsurfaces = 20 if forplotting else 8
target_areas = np.asarray([1.0 + i*1.0 for i in range(nsurfaces)])
# target_fluxes = np.asarray([((i+1)*2.650e-02)**2 for i in range(nsurfaces)])
l = 0.0167
for i in range(nsurfaces):
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, quadpoints_phi=phis, quadpoints_theta=thetas)
    if sold is None:
        s.fit_to_curve(ma, l, flip_theta=True)
    else:
        s.set_dofs(sold.get_dofs())
        s.extend_via_normal(-l)

    tf = ToroidalFlux(s, bs_tf)
    ar = Area(s)
    # ar_target = ar.J()
    print('Area', ar.J())
    ar_target = target_areas[i]
    boozer_surface = BoozerSurface(bs, s, ar, ar_target) 
    # boozer_surface = BoozerSurface(bs, s, tf, tf_target) 

    # compute surface first using LBFGS exact and an area constraint
    if i == 0:
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=3000, constraint_weight=100., iota=iota, G=G)
        print(f"iota={res['iota']:.3f}, sqrt(tf)={np.sqrt(tf.J()):.3e}, area={ar.J():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    # s.plot(show=True)
    # else:
    #     s, iota, _ = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=300, constraint_weight=100., iota=iota)
    #     print(f"iota={iota:.3f}, tf={tf.J():.3f}, area={ar.J():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, iota, bs, derivatives=0)):.3e}")
    # s.plot()
    res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
    print(f"iota={res['iota']:.3f}, sqrt(tf)={np.sqrt(tf.J()):.3e}, area={ar.J():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    print(s.gamma()[0, 0, :])
    surfaces_is.append(s)
    allres.append(res)
    sold = s

surfaces_oos = []
surfaces_oos_full = []
surfaces_is_full = []
# phis_full = np.linspace(0, 1, 2*(nfp*ntor)+1, endpoint=false)
# thetas_full = np.linspace(0, 1, 2*(mpol)+1, endpoint=false)
phis_full = np.linspace(0, 1, 2*(nfp*ntor)+1, endpoint=false)
thetas_full = np.linspace(0, 1, 2*(mpol)+1, endpoint=false)
for i in range(nsurfaces):
    surfaces_oos.append(SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, quadpoints_phi=phis, quadpoints_theta=thetas))

    surfaces_oos_full.append(SurfaceXYZTensorFourier(
        mpol=mpol, ntor=nfp*ntor, nfp=1, stellsym=False, clamped_dims=[False, True, False], quadpoints_phi=phis_full, quadpoints_theta=thetas_full))
    surfaces_is_full.append(SurfaceXYZTensorFourier(
        mpol=mpol, ntor=nfp*ntor, nfp=1, stellsym=False, clamped_dims=[False, True, False], quadpoints_phi=phis_full, quadpoints_theta=thetas_full))




swork = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, nfp=nfp, stellsym=True, quadpoints_phi=phis_full, quadpoints_theta=thetas_full)
sfullwork = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=nfp*ntor, nfp=1, stellsym=False, clamped_dims=[False, True, False], quadpoints_phi=phis_full, quadpoints_theta=thetas_full)

def partial_to_full(spartial, sfull):
    swork.set_dofs(spartial.get_dofs())
    sfullwork.least_squares_fit(swork.gamma())
    sfull.set_dofs(sfullwork.get_dofs())


def compute_surface(bs_pert, res, s, ar_target, bfgs_first=0, exact=False):
    ar = Area(s)
    # bs_flux = BiotSavart(bs_pert.coils, bs_pert.coil_currents)
    boozer_surface = BoozerSurface(bs_pert, s, ar, ar_target) 
    if bfgs_first > 0:
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=bfgs_first, constraint_weight=100., iota=res['iota'], G=res['G'])
        print(f"iota={res['iota']:.10f}, tf={ToroidalFlux(s, bs_pert).J():.3e}, area={ar.J():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs_pert, derivatives=0)):.3e}, ||gradient||={np.linalg.norm(res['gradient']):.3e}")
    res = boozer_surface.minimize_boozer_penalty_constraints_ls(
        tol=1e-10, maxiter=100, constraint_weight=1e2, iota=res['iota'], G=res['G'], method='manual', linear_solver='lu', lam=1e-4)
    print(f"iota={res['iota']:.10f}, tf={ToroidalFlux(s, bs_pert).J():.3e}, area={ar.J():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs_pert, derivatives=0)):.3e}, ||gradient||={np.linalg.norm(res['gradient']):.3e}, iter={res['iter']}")
    if exact:
        res = boozer_surface.solve_residual_equation_exactly_newton(iota=res['iota'], G=res['G'])
        print(f"iota={res['iota']:.10f}, tf={ToroidalFlux(s, bs_pert).J():.3e}, area={ar.J():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs_pert, derivatives=0)):.3e}")
        if np.linalg.norm(res['residual']) > 1e-9:
            raise RuntimeError('norm of residual too large')
    else:
        if np.linalg.norm(res['gradient']) > 1e-8:
            raise RuntimeError('norm of gradient too large')
    return res



def compute_non_quasisymmetry_l2(s, bs):
    x = s.gamma()
    B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
    mod_B = np.linalg.norm(B, axis=2)
    n = np.linalg.norm(s.normal(), axis=2)
    mean_phi_mod_B = np.mean(mod_B, axis=0)
    mod_B_QS = mean_phi_mod_B[None, :]
    mod_B_non_QS = mod_B - mod_B_QS
    non_qs = np.mean(mod_B_non_QS**2)
    qs = np.mean(mod_B_QS**2)
    return non_qs, qs

def compute_non_quasisymmetry_L2(s, bs):
    x = s.gamma()
    B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
    mod_B = np.linalg.norm(B, axis=2)
    n = np.linalg.norm(s.normal(), axis=2)
    mean_phi_mod_B = np.mean(mod_B*n, axis=0)/np.mean(n, axis=0)
    mod_B_QS = mean_phi_mod_B[None, :]
    mod_B_non_QS = mod_B - mod_B_QS
    non_qs = np.mean(mod_B_non_QS**2 * n)
    qs = np.mean(mod_B_QS**2 * n)
    return non_qs, qs


# non_qs = [compute_non_quasisymmetry(surfaces_full[j], bs) for j in range(nsurfaces)]
# print(non_qs)
res_is = []
for i in range(nsurfaces):
    partial_to_full(surfaces_is[i], surfaces_is_full[i])
    try:
        res_is.append(compute_surface(bs, allres[i], surfaces_is_full[i], target_areas[i]))
        plot(s, savedir + f"plot_is_{i}.png")
    except Exception as ex:
        print(ex)
        break

nsurfaces_success = len(res_is)
surfaces_is = [r['s'] for r in res_is]
is_non_qs_L2 = np.full((nsurfaces, ), np.nan)
is_non_qs_l2 = np.full((nsurfaces, ), np.nan)
is_qs_L2 = np.full((nsurfaces, ), np.nan)
is_qs_l2 = np.full((nsurfaces, ), np.nan)
is_iotas = np.full((nsurfaces, ), np.nan)
surface_dofs = [r['s'].get_dofs() for r in res_is]
if forplotting:
    np.save(outdir + "/surfacedofsforplotting.npy", surface_dofs)
    import sys; sys.exit()
else:
    np.save(outdir + "/surfacedofs.npy", surface_dofs)
for j in range(nsurfaces_success):
    is_iotas[j] = res_is[j]['iota']
    is_non_qs_L2[j], is_qs_L2[j] = compute_non_quasisymmetry_L2(res_is[j]['s'], bs)
    is_non_qs_l2[j], is_qs_l2[j] = compute_non_quasisymmetry_l2(res_is[j]['s'], bs)

print("is_iotas", is_iotas)
print("is_non_qs_L2", is_non_qs_L2)
print("is_qs_L2", is_qs_L2)
np.save(savedir + "/is_non_qs_L2.npy", is_non_qs_L2)
np.save(savedir + "/is_non_qs_l2.npy", is_non_qs_l2)
np.save(savedir + "/is_qs_L2.npy", is_qs_L2)
np.save(savedir + "/is_qs_l2.npy", is_qs_l2)
np.save(savedir + "/is_iotas.npy", is_iotas)
obj.compute_out_of_sample()
N = obj.noutsamples
oos_non_qs_L2 = np.full((nsurfaces, N), np.nan)
oos_non_qs_l2 = np.full((nsurfaces, N), np.nan)
oos_qs_L2 = np.full((nsurfaces, N), np.nan)
oos_qs_l2 = np.full((nsurfaces, N), np.nan)
oos_iotas = np.full((nsurfaces, N), np.nan)
for i in range(N):
    print(f"Sample {i}")
    bs_pert = obj.stochastic_qs_objective_out_of_sample.J_BSvsQS_perturbed[i].biotsavart
    ma_points = (2*(nfp*ntor)+1) * 10
    madata_pert = find_magnetic_axis(bs_pert, ma_points, np.linalg.norm(ma.gamma()[0, 0:2]), output='cartesian')
    ma_pert = CurveRZFourier(ma_points, ma.order, 1, False)
    ma_pert.least_squares_fit(madata_pert)
    sample_origs = [[s.copy() for s in c.sample] for c in bs_pert.coils]
    ncont = 10
    alphas = np.linspace(0, 1., ncont, endpoint=True)
    for j in range(nsurfaces_success):
        print(f"Surface {j}")
        # if j == 0:
        #     s = surfaces_oos[j]
        #     s.fit_to_curve(ma, l, flip_theta=True)
        # else:
        #     s = surfaces_oos[j]
        #     s.set_dofs(surfaces_oos[j-1].get_dofs())
        #     s.extend_via_normal(-l)
        try:
            # res_oos = compute_surface(bs_pert, res_is[j], s, target_areas[j], bfgs_first=1000 if (j==0) else 20, exact=False)
            # partial_to_full(s, sfull)
            # print(sfull.gamma()[0, 0, :])
            if j == 0:
                sfull = surfaces_oos_full[j]
                res_oos = res_is[j]
                sfull.set_dofs(surfaces_is_full[j].get_dofs())
                for k in range(ncont):
                    for ci in range(len(bs_pert.coils)):
                        bs_pert.coils[ci].sample = [alphas[k] * s for s in sample_origs[ci]]
                        bs_pert.coils[ci].invalidate_cache()
                    res_oos = compute_surface(bs_pert, res_oos, sfull, target_areas[j], bfgs_first=False, exact=True)
            else:
                s = surfaces_oos[j]
                s.set_dofs(surfaces_oos[j-1].get_dofs())
                s.extend_via_normal(-l)
                res_oos = compute_surface(bs_pert, res_oos, sfull, target_areas[j], bfgs_first=20, exact=False)
            oos_non_qs_L2[j, i], oos_qs_L2[j, i] = compute_non_quasisymmetry_L2(res_oos['s'], bs_pert)
            oos_non_qs_l2[j, i], oos_qs_l2[j, i] = compute_non_quasisymmetry_l2(res_oos['s'], bs_pert)
            oos_iotas[j, i] = res_oos['iota']
            plot(surfaces_oos[j], savedir + f"plot_oos_{j}_{i}.png")
        except Exception as ex:
            print(ex)
            break


np.set_printoptions(edgeitems=30, linewidth=200, formatter=dict(float=lambda x: "%.5e" % x))
print("Non-QS    ", is_non_qs_L2)
print("Non-QS OOS", np.mean(oos_non_qs_L2, axis=1))
print("Non-QS OOS", oos_non_qs_L2)
print("oos_iotas", oos_iotas)
print("is_iotas", is_iotas)

np.save(savedir + "/oos_non_qs_L2.npy", oos_non_qs_L2)
np.save(savedir + "/oos_non_qs_l2.npy", oos_non_qs_l2)
np.save(savedir + "/oos_qs_L2.npy", oos_qs_L2)
np.save(savedir + "/oos_qs_l2.npy", oos_qs_l2)
np.save(savedir + "/oos_iotas.npy", oos_iotas)
