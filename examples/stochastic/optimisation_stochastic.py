from pyplasmaopt import *
from objective_stochastic import stochastic_get_objective
from scipy.optimize import minimize
from lbfgs import fmin_lbfgs

import numpy as np
import os

obj, args = stochastic_get_objective()
# import sys; sys.exit()

outdir = obj.outdir


def taylor_test(obj, x, order=6):
    np.random.seed(1)
    h = np.random.rand(*(x.shape))
    obj.update(x)
    dj0 = obj.dres
    djh = sum(dj0*h)
    if order == 1:
        shifts = [0, 1]
        weights = [-1, 1]
    elif order == 2:
        shifts = [-1, 1]
        weights = [-0.5, 0.5]
    elif order == 4:
        shifts = [-2, -1, 1, 2]
        weights = [1/12, -2/3, 2/3, -1/12]
    elif order == 6:
        shifts = [-3, -2, -1, 1, 2, 3]
        weights = [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]
    for i in range(10, 40):
        eps = 0.5**i
        obj.update(x + shifts[0]*eps*h)
        fd = weights[0] * obj.res
        for i in range(1, len(shifts)):
            obj.update(x + shifts[i]*eps*h)
            fd += weights[i] * obj.res
        err = abs(fd/eps - djh)
        info("%.6e, %.6e, %.6e", eps, err, err/np.linalg.norm(djh))
    obj.update(x)
    info("-----")



# For the CVaR optimisation we use the result of stochastic optimisation as the initial guess.
if obj.mode == "cvar":
    try:
        x = np.concatenate((np.load(outdir.replace(args.mode.replace(".", "p"), "stochastic") + "xmin.npy"), [0.]))
        info('Found initial guess from stochastic optimization.')
    except:
        warning('Could not find initial guess from stochastic optimization.')
        x = obj.x0
else:
    x = obj.x0

#x = np.load("output/ncsx_atopt_mode-stochastic_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-4096_noutsamples-4096_seed-1_sigma-0p001_length_scale-0p2/xmin.npy")

if obj.mode == "cvar":
    obj.update(x)
    x[-1] = obj.cvar.find_optimal_t(obj.Jsamples ,x[-1])
obj.update(x)
obj.callback(x)
#obj.save_to_matlab('matlab_init')
if False:
    taylor_test(obj, x, order=1)
    taylor_test(obj, x, order=2)
    taylor_test(obj, x, order=4)
    taylor_test(obj, x, order=6)

maxiter = 15000

def J_scipy(x):
    try:
        obj.update(x)
        return obj.res, obj.dres
    except RuntimeError as ex:
        return 2*obj.res, -obj.dres
def J_pylbfgs(x, g, *args):
    try:
        obj.update(x)
        g[:] = obj.dres
        return obj.res
    except RuntimeError as ex:
        g[:] = -obj.dres
        return 2*obj.res

xmin = [x.copy()]
def p_pylbfgs(x, *args):
    obj.callback(x)
    xmin[0][:] = x
    return 0

import time
t1 = time.time()
iters = 0
restarts = 0
while iters < maxiter and restarts < 30:
    if iters > 0:
        info("####################################################################################################")
        info("################################# Restart optimization #############################################")
        info("####################################################################################################")
        restarts += 1
    if obj.mode == "cvar" and restarts < 6:
        miter = min(1000, maxiter-iters)
    else:
        miter = min(10000, maxiter-iters)
    if args.optim == 'pylbfgs':
        try:
            res = fmin_lbfgs(J_pylbfgs, x, progress=p_pylbfgs, max_iterations=miter, m=500, line_search='wolfe', max_linesearch=40, epsilon=1e-12)
        except Exception as e:
            info(e)
            pass
        x = xmin[0].copy()
        iters = len(obj.Jvals)
    else:
        res = minimize(J_scipy, x, jac=True, method='bfgs', tol=1e-20, options={"maxiter": miter}, callback=obj.callback)
        iters += res.nit
        x = res.x

    if obj.mode == "cvar" and restarts < 6:
        obj.cvar.eps *= 0.1**0.5
        x[-1] = obj.cvar.find_optimal_t(obj.Jsamples ,x[-1])

t2 = time.time()
info(res)
if args.optim == "pylbfgs":
    xmin = x
else:
    info(f"Time per iteration: {(t2-t1)/res.nfev}")
    info(f"Gradient norm at minimum: {np.linalg.norm(res.jac)}")
    xmin = res.x

J_distance = MinimumDistance(obj.stellarator.coils, 0)
info("Minimum distance = %f" % J_distance.min_dist())
# import IPython; IPython.embed()
# import sys; sys.exit()
# obj.save_to_matlab('matlab_optim')
# obj.stellarator.savetotxt(outdir)
# matlabcoils = [c.tomatlabformat() for c in obj.stellarator._base_coils]
# np.savetxt(os.path.join(obj.outdir, 'coilsmatlab.txt'), np.hstack(matlabcoils))
# np.savetxt(os.path.join(obj.outdir, 'currents.txt'), obj.stellarator._base_currents)
if comm.rank == 0:
    np.save(outdir + "xmin.npy", xmin)
    np.save(outdir + "Jvals.npy", obj.Jvals)
    np.save(outdir + "dJvals.npy", obj.dJvals)
    np.save(outdir + "Jvals_quantiles.npy", obj.Jvals_quantiles)
    np.save(outdir + "Jvals_no_noise.npy", obj.Jvals_no_noise)
    np.save(outdir + "xiterates.npy", obj.xiterates)
    np.save(outdir + "Jvals_individual.npy", obj.Jvals_individual)
    np.save(outdir + "Jvals_perturbed.npy", obj.Jvals_perturbed)
    np.save(outdir + "QSvsBS_perturbed.npy", obj.QSvsBS_perturbed)
    np.save(outdir + "Jvals_insample.npy", obj.Jvals_perturbed[-1])
    np.save(outdir + "QSvsBS_insample.npy", obj.QSvsBS_perturbed[-1])
    if args.noutsamples > 0:
        np.save(outdir + "out_of_sample_values.npy", obj.out_of_sample_values)
        np.save(outdir + "out_of_sample_means.npy", np.mean(obj.out_of_sample_values, axis=1))

def approx_H(x):
    n = x.size
    H = np.zeros((n, n))
    x0 = x
    eps = 1e-4
    for i in range(n):
        x = x0.copy()
        x[i] += eps
        d1 = J_scipy(x)[1]
        x[i] -= 2*eps
        d0 = J_scipy(x)[1]
        H[i, :] = (d1-d0)/(2*eps)
    H = 0.5 * (H+H.T)
    return H

from scipy.linalg import eigh
H = approx_H(x)
D, E = eigh(H)
info('evals: %s', D)
if comm.rank == 0:
    np.save(outdir + "evals_opt.npy", D)

info('Final out of samples computation')
oos = []
oosloops = 256
for i in range(oosloops):
    info(f"{i}/{oosloops}")
    oos.append(obj.compute_out_of_sample())
    obj.stochastic_qs_objective_out_of_sample.resample()

QSvsBS_outofsample = np.concatenate([o[0] for o in oos])
Jvals_outofsample = np.concatenate([o[1] for o in oos])

if comm.rank == 0:
    np.save(outdir + "QSvsBS_outofsample.npy", QSvsBS_outofsample)
    np.save(outdir + "Jvals_outofsample.npy", Jvals_outofsample)

if True:
    taylor_test(obj, xmin, order=4)
