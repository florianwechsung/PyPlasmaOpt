from pyplasmaopt import *
from objective_stochastic import stochastic_get_objective
from scipy.optimize import minimize
import numpy as np
import os

obj, args = stochastic_get_objective()
obj.plot('tmp.png')
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
        x = np.concatenate((np.loadtxt(outdir.replace(args.mode.replace(".", "p"), "stochastic") + "xmin.txt"), [0.]))
    except:
        x = obj.x0
else:
    x = obj.x0

if obj.mode == "cvar":
    obj.update(x)
    x[-1] = obj.cvar.find_optimal_t(obj.QSvsBS_perturbed[-1] ,x[-1])
obj.update(x)
obj.callback(x)
#obj.save_to_matlab('matlab_init')
if False:
    taylor_test(obj, x, order=1)
    taylor_test(obj, x, order=2)
    taylor_test(obj, x, order=4)
    taylor_test(obj, x, order=6)

maxiter = 5000 if obj.mode == "cvar" else 5000

def J_scipy(x):
    obj.update(x)
    res = obj.res
    dres = obj.dres
    return res, dres
import time
t1 = time.time()
iters = 0
restarts = 0
while iters < maxiter and restarts < 10:
    if iters > 0:
        if comm.rank == 0:
            info("####################################################################################################")
            info("################################# Restart optimization #############################################")
            info("####################################################################################################")
        restarts += 1
    if obj.mode == "cvar": 
        miter = min(500, maxiter-iters)
        res = minimize(J_scipy, x, jac=True, method='bfgs', tol=1e-20, options={"maxiter": miter}, callback=obj.callback)
        obj.cvar.eps *= 0.1
        x[-1] = obj.cvar.find_optimal_t(obj.QSvsBS_perturbed[-1] ,x[-1])
    else:
        miter = maxiter-iters
        res = minimize(J_scipy, x, jac=True, method='bfgs', tol=1e-20, options={"maxiter": miter}, callback=obj.callback)

    iters += res.nit
    x = res.x

t2 = time.time()
if comm.rank == 0:
    info(res)
    info(f"Time per iteration: {(t2-t1)/len(obj.Jvals)}")
    info(f"Gradient norm at minimum: {np.linalg.norm(res.jac)}")

xmin = res.x

info("%s" % res)
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
    np.savetxt(outdir + "xmin.txt", xmin)
    np.savetxt(outdir + "Jvals.txt", obj.Jvals)
    np.savetxt(outdir + "dJvals.txt", obj.dJvals)
    np.savetxt(outdir + "Jvals_quantiles.txt", obj.Jvals_quantiles)
    np.savetxt(outdir + "Jvals_no_noise.txt", obj.Jvals_no_noise)
    np.savetxt(outdir + "xiterates.txt", obj.xiterates)
    np.savetxt(outdir + "Jvals_individual.txt", obj.Jvals_individual)
    np.savetxt(outdir + "Jvals_insample.txt", obj.Jvals_perturbed)
    np.savetxt(outdir + "QSvsBS_insample.txt", obj.QSvsBS_perturbed)
    if args.noutsamples > 0:
        np.savetxt(outdir + "out_of_sample_values.txt", obj.out_of_sample_values)
        np.savetxt(outdir + "out_of_sample_means.txt", np.mean(obj.out_of_sample_values, axis=1))

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

# from scipy.linalg import eigh
# H = approx_H(x)
# D, E = eigh(H)
# info('evals: %s', D)
# if comm.rank == 0:
#     np.savetxt(outdir + "evals_opt.txt", D)

info('Final out of samples computation')
oos = []
for i in range((2**15)//args.noutsamples):
    info(f"{i*args.noutsamples} / {2**15}")
    oos.append(obj.compute_out_of_sample())
    obj.stochastic_qs_objective_out_of_sample.resample()

QSvsBS_outofsample = np.concatenate([o[0] for o in oos])
Jvals_outofsample = np.concatenate([o[1] for o in oos])

if comm.rank == 0:
    np.savetxt(outdir + "QSvsBS_outofsample.txt", QSvsBS_outofsample)
    np.savetxt(outdir + "Jvals_outofsample.txt", Jvals_outofsample)

if True:
    taylor_test(obj, xmin, order=4)
