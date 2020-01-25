from pyplasmaopt import *
from problem2_objective import Problem2_Objective, get_objective
import numpy as np
import matplotlib.pyplot as plt
import os

solver = "scipy"
# solver = "nlopt"
# solver = "pylbfgs"
# solver = None
obj, args = get_objective()

print("Biggest deviation", np.max(np.linalg.norm(obj.J_BSvsQS_perturbed[0].biotsavart.coils[0].sample[0], axis=1))*1e3, "mm")

print(obj.x0.shape)
info_dict = {'Nfeval': 0}
outdir = obj.outdir
os.makedirs(outdir, exist_ok=True)

def taylor_test(obj, x):
    obj.update(x)
    J0, dJ0 = obj.res, obj.dres
    np.random.seed(1)
    h = 0.1 * np.random.rand(*(x.shape))
    dJh = sum(dJ0*h)
    for i in range(1, 8):
        eps = 0.1**i
        shifts = [-3, -2, -1, 1, 2, 3]
        weights = [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]

        obj.update(x + shifts[0]*eps*h)
        fd = weights[0] * obj.res
        for i in range(1, len(shifts)):
            obj.update(x + shifts[i]*eps*h)
            fd += weights[i] * obj.res
        err = abs(fd/eps - dJh)
        print(err, err/np.linalg.norm(dJh))
    obj.update(x)

x = obj.x0
obj.update(x)
obj.callback(x)
# import IPython; IPython.embed()
# import sys; sys.exit()

if True:
    taylor_test(obj, x)
    # import sys
    # sys.exit()

maxiter = 1000
memory = 300
if solver == "nlopt":
    def J_nlopt(x, grad, info=info_dict):
        obj.update(x)
        if grad.size > 0:
            grad[:] = obj.dres[:]
        if info:
            plot(info)
        return obj.res
    import nlopt
    opt = nlopt.opt(nlopt.LD_LBFGS, len(obj.x0))
    opt.set_min_objective(J_nlopt)
    opt.set_xtol_rel(1e-8)
    opt.set_vector_storage(memory)
    opt.set_maxeval(maxiter)
    xmin = opt.optimize(list(x))
elif solver == "scipy":
    def J_scipy(x, info=None):
        obj.update(x)
        res = obj.res
        dres = obj.dres
        return res, dres
    import time
    from scipy.optimize import minimize
    t1 = time.time()
    iters = 0
    restarts = 0
    while iters < maxiter and restarts < 10:
        if iters > 0:
            print("####################################################################################################")
            print("################################# Restart optimization #############################################")
            print("####################################################################################################")
            restarts += 1
        # res = minimize(J_scipy, x, args=(info_dict,), jac=True, method='L-BFGS-B', tol=1e-20, options={'maxiter': maxiter-iters, 'maxcor': memory}, callback=obj.callback)
        res = minimize(J_scipy, x, args=(info_dict,), jac=True, method='BFGS', tol=1e-20, options={'maxiter': maxiter-iters}, callback=obj.callback)
        iters += res.nit
        x = res.x

    t2 = time.time()
    print(f"Time per iteration: {(t2-t1)/len(obj.Jvals)}")
    print(res)
    xmin = res.x
    print("Gradient norm at minimum:", np.linalg.norm(res.jac))
elif solver == "pylbfgs":
    from lbfgs import LBFGS
    def J_pylbfgs(x, g, info=info_dict):
        obj.update(x)
        g[:] = obj.dres[:]
        plot(info)
        return obj.res

    opt = LBFGS()
    opt.max_iterations = maxiter
    opt.linesearch = "wolfe"

    xmin = opt.minimize(J_pylbfgs, x)
else:
    xmin = np.loadtxt(outdir + "xmin.txt")
    obj.update(xmin)

np.savetxt(outdir + "xmin.txt", xmin)
np.savetxt(outdir + "Jvals.txt", obj.Jvals)
np.savetxt(outdir + "dJvals.txt", obj.dJvals)
np.savetxt(outdir + "Jvals_quantiles.txt", obj.Jvals_quantiles)
np.savetxt(outdir + "Jvals_no_noise.txt", obj.Jvals_no_noise)
np.savetxt(outdir + "xiterates.txt", obj.xiterates)
np.savetxt(outdir + "Jvals_perturbed.txt", obj.Jvals_perturbed)

# import IPython; IPython.embed()
if True:
    taylor_test(obj, xmin)
# import sys
# sys.exit()


stellarator = obj.stellarator
if True:
    perturbed_coils = [GaussianPerturbedCurve(coil, obj.sampler) for coil in stellarator.coils]
    perturbed_bs = BiotSavart(perturbed_coils, stellarator.currents)
    perturbed_bs.set_points(obj.ma.gamma)
    J_BSvsQS = BiotSavartQuasiSymmetricFieldDifference(obj.qsf, perturbed_bs)

    L2s = [0.5 * obj.J_BSvsQS.J_L2()]
    H1s = [0.5 * obj.J_BSvsQS.J_H1()]
    Jvals_perturbed_more = [obj.res - obj.res1 + L2s[-1] + H1s[-1]]
    print(L2s[0], H1s[0])
    nsamples = 2000
    for i in range(nsamples):
        if i % 100 == 0:
            print(i, flush=True)
        [coil.resample() for coil in perturbed_coils]
        perturbed_bs.clear_cached_properties()
        L2s.append(0.5 * J_BSvsQS.J_L2())
        H1s.append(0.5 * J_BSvsQS.J_H1())
        Jvals_perturbed_more.append(obj.res - obj.res1 + L2s[-1] + H1s[-1])
    np.savetxt(outdir + "L2s.txt", L2s)
    np.savetxt(outdir + "H1s.txt", H1s)
    np.savetxt(outdir + "Jvals_perturbed_more.txt", Jvals_perturbed_more)
# import IPython; IPython.embed()
