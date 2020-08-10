from pyplasmaopt import *
from example3_get_objective import example3_get_objective
from scipy.optimize import minimize
import numpy as np
import os

obj, args = example3_get_objective()
obj.plot('tmp.png')
# import sys; sys.exit()

outdir = obj.outdir


def taylor_test(obj, x, order=6):
    obj.update(x)
    dj0 = obj.dres
    np.random.seed(1)
    h = np.random.rand(*(x.shape))
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



x = obj.x0
obj.update(x)
obj.callback(x)
# matlabcoils = [c.tomatlabformat() for c in obj.stellarator._base_coils]
# np.savetxt(os.path.join(obj.outdir, 'coilsmatlab_init.txt'), np.hstack(matlabcoils))

# if True:
#     taylor_test(obj, x, order=1)
#     taylor_test(obj, x, order=2)
    # taylor_test(obj, x, order=4)
    # taylor_test(obj, x, order=6)

maxiter = 5000
memory = 200


def J_scipy(x):
    obj.update(x)
    res = obj.res
    dres = obj.dres
    return res, dres


res = minimize(J_scipy, x, jac=True, method='l-bfgs-b', tol=1e-20,
               options={"maxiter": maxiter, "maxcor": memory},
               callback=obj.callback)
print(res)
xmin = res.x
J_distance = MinimumDistance(obj.stellarator.coils, 0)
info("Minimum distance = %f" % J_distance.min_dist())
obj.stellarator.savetotxt(outdir)
matlabcoils = [c.tomatlabformat() for c in obj.stellarator._base_coils]
np.savetxt(os.path.join(obj.outdir, 'coilsmatlab.txt'), np.hstack(matlabcoils))
np.savetxt(os.path.join(obj.outdir, 'currents.txt'), obj.stellarator._base_currents)
import IPython; IPython.embed()

# def approx_H(x):
#     n = x.size
#     H = np.zeros((n, n))
#     x0 = x
#     d0 = J_scipy(x0)[1]
#     eps = 1e-6
#     for i in range(n):
#         x1 = x0.copy()
#         x1[i] += eps
#         d1 = J_scipy(x1)[1]
#         H[i, :] = (d1-d0)/eps
#     H = 0.5 * (H+H.T)
#     return H

# from scipy.linalg import eigh
# x = xmin
# for i in range(20):
#     H = approx_H(x)
#     f, d = J_scipy(x)
#     print(f, np.linalg.norm(d))
#     D, E = eigh(H)
#     D = np.abs(D)
#     # D = np.maximum(D, 0.)
#     evals_sorted = np.sort(D)
#     D[D < evals_sorted[20]] = 0.
#     D += 1e-1
#     s = E @ np.diag(1./D) @ E.T @ d
#     alpha = 0.3
#     x -= alpha * s


# import IPython; IPython.embed()

np.savetxt(outdir + "xmin.txt", xmin)
np.savetxt(outdir + "Jvals.txt", obj.Jvals)
np.savetxt(outdir + "dJvals.txt", obj.dJvals)
np.savetxt(outdir + "xiterates.txt", obj.xiterates)
np.savetxt(outdir + "Jvals_individual.txt", obj.Jvals_individual)

if True:
    taylor_test(obj, xmin)
