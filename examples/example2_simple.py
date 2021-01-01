from pyplasmaopt import *
from example2_get_objective import example2_get_objective
from scipy.optimize import minimize
import numpy as np

obj, args = example2_get_objective()

outdir = obj.outdir
solver = args.optimizer.lower()
assert solver in ["bfgs", "lbfgs", "l-bfgs-b"]
if solver == "lbfgs":
    solver = "l-bfgs-b"


def taylor_test(obj, x):
    obj.update(x)
    dj0 = obj.dres
    np.random.seed(1)
    h = 0.1 * np.random.rand(*(x.shape))
    djh = sum(dj0*h)
    for i in range(1, 15):
        eps = 0.5**i
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


x = obj.x0
obj.update(x)
obj.callback(x)

if False:
    taylor_test(obj, x)

maxiter = 2000
memory = 200


def J_scipy(x):
    obj.update(x)
    res = obj.res
    dres = obj.dres
    return res, dres

res = minimize(J_scipy, x, jac=True, method=solver, tol=1e-20,
               options={"maxiter": maxiter, "maxcor": memory},
               callback=obj.callback)
info("%s" % res)
xmin = res.x

# self.J_distance = MinimumDistance(stellarator.coils, minimum_distance)

def approx_H(x):
    idxs = obj.coil_dof_idxs
    n = idxs[1] - idxs[0]
    H = np.zeros((n, n))
    x0 = x
    eps = 1e-5
    for i, idx in enumerate(range(*idxs)):
        x1 = x0.copy()
        x1[idx] += eps
        d1 = J_scipy(x1)[1]
        x1[idx] -= 2*eps
        d2 = J_scipy(x1)[1]
        H[i, :] = ((d1-d2)/(2*eps))[idxs[0]:idxs[1]]
    H = 0.5 * (H+H.T)
    return H

from scipy.linalg import eigh
H = approx_H(xmin)
D, E = eigh(H)
D = np.sort(np.abs(D))
print(D)
import matplotlib.pyplot as plt
plt.semilogy(D)
plt.savefig(obj.outdir + "/evals.png")
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
