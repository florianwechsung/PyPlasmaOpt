from pyplasmaopt import *
from problem2_objective import Problem2_Objective, get_objective
from scipy.optimize import minimize
import numpy as np
import os

obj, args = get_objective()

info("Biggest deviation in first coil %.6fmm", np.max(np.linalg.norm(obj.stochastic_qs_objective.J_BSvsQS_perturbed[0].biotsavart.coils[0].sample[0], axis=1))*1e3)

outdir = obj.outdir
solver = args.optimizer
# solver = None

def taylor_test(obj, x):
    obj.update(x)
    j0, dj0 = obj.res, obj.dres
    np.random.seed(1)
    h = 0.1 * np.random.rand(*(x.shape))
    djh = sum(dj0*h)
    for i in range(1, 8):
        eps = 0.1**i
        shifts = [-3, -2, -1, 1, 2, 3]
        weights = [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]

        obj.update(x + shifts[0]*eps*h)
        fd = weights[0] * obj.res
        for i in range(1, len(shifts)):
            obj.update(x + shifts[i]*eps*h)
            fd += weights[i] * obj.res
        err = abs(fd/eps - djh)
        info("%.6e, %.6e", err, err/np.linalg.norm(djh))
    obj.update(x)


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
# import IPython; IPython.embed()
# import sys; sys.exit()

if True:
    taylor_test(obj, x)
    # import sys; sys.exit()

maxiter = 25
memory = 200
if solver is None:
    xmin = np.loadtxt(outdir + "xmin.txt")
    obj.update(xmin)
    obj.callback(xmin)
elif solver.lower() in ["bfgs", "lbfgs"]:
    method = {"bfgs": "BFGS", "lbfgs": "L-BFGS-B"}[solver]
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
            miter = min(250, maxiter-iters)
            res = minimize(J_scipy, x, jac=True, method=method, tol=1e-20, options={"maxiter": miter, "maxcor": memory}, callback=obj.callback)
            obj.cvar.eps *= 0.1
            x[-1] = obj.cvar.find_optimal_t(obj.QSvsBS_perturbed[-1] ,x[-1])
        else:
            miter = maxiter-iters
            res = minimize(J_scipy, x, jac=True, method=method, tol=1e-20, options={"maxiter": miter, "maxcor": memory}, callback=obj.callback)

        iters += res.nit
        x = res.x

    t2 = time.time()
    if comm.rank == 0:
        info(res)
        info(f"Time per iteration: {(t2-t1)/len(obj.Jvals)}")
        info(f"Gradient norm at minimum: {np.linalg.norm(res.jac)}")
    xmin = res.x
elif solver.lower() in ["sgd"]:
    def J_scipy(x):
        obj.update(x)
        res = obj.res
        dres = obj.dres
        return res, dres
    # oldmode = obj.mode
    # obj.mode = 'deterministic'
    res = minimize(J_scipy, x, jac=True, method="BFGS", tol=1e-20, options={"maxiter": 75, "maxcor": 1000}, callback=obj.callback)
    # obj.mode = oldmode

    learning_rate = args.lr
    x = res.x
    def J(x):
        # obj.stochastic_qs_objective.resample()
        obj.update(x)
        res = obj.res
        dres = obj.dres
        return res, dres
    P = res.hess_inv
    # P = None
    import time
    t1 = time.time()
    xmin = gradient_descent(J, x, learning_rate, maxiter, callback=obj.callback, P=P)
    # xmin = ada_grad(J, x, learning_rate, maxiter, callback=obj.callback, P=P)
    # xmin = momentum(J, x, learning_rate, maxiter, callback=obj.callback, P=P)
    # xmin = rmsprop(J, x, learning_rate, maxiter, callback=obj.callback, P=P)
    t2 = time.time()

if comm.rank == 0 and solver is not None:
    np.savetxt(outdir + "xmin.txt", xmin)
    np.savetxt(outdir + "Jvals.txt", obj.Jvals)
    np.savetxt(outdir + "dJvals.txt", obj.dJvals)
    np.savetxt(outdir + "Jvals_quantiles.txt", obj.Jvals_quantiles)
    np.savetxt(outdir + "Jvals_no_noise.txt", obj.Jvals_no_noise)
    np.savetxt(outdir + "xiterates.txt", obj.xiterates)
    np.savetxt(outdir + "Jvals_individual.txt", obj.Jvals_individual)
    np.savetxt(outdir + "Jvals_insample.txt", obj.Jvals_perturbed)
    np.savetxt(outdir + "QSvsBS_insample.txt", obj.QSvsBS_perturbed)
    np.savetxt(outdir + "out_of_sample_values.txt", obj.out_of_sample_values)
    np.savetxt(outdir + "out_of_sample_means.txt", np.mean(obj.out_of_sample_values, axis=1))

# import IPython; IPython.embed()
# import sys; sys.exit()

if True:
    taylor_test(obj, xmin)

QSvsBS_outofsample, Jvals_outofsample = obj.compute_out_of_sample()
if comm.rank == 0:
    np.savetxt(outdir + "QSvsBS_outofsample.txt", QSvsBS_outofsample)
    np.savetxt(outdir + "Jvals_outofsample.txt", Jvals_outofsample)
