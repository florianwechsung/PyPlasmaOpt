from pyplasmaopt import *
from problem2_objective import get_objective
from scipy.optimize import minimize
import numpy as np

obj, args = get_objective()

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


x = obj.x0
obj.update(x)
obj.callback(x)

if True:
    taylor_test(obj, x)

maxiter = 2000 #maximum number of optimization iterations
memory = 200


def J_scipy(x):
    obj.update(x)
    res = obj.res
    dres = obj.dres
    return res, dres


res = minimize(J_scipy, x, jac=True, method=solver, tol=1e-20,
               options={"maxiter": maxiter, "maxcor": memory},
               callback=obj.callback)
xmin = res.x

np.savetxt(outdir + "xmin.txt", xmin)
np.savetxt(outdir + "Jvals.txt", obj.Jvals)
np.savetxt(outdir + "dJvals.txt", obj.dJvals)
np.savetxt(outdir + "xiterates.txt", obj.xiterates)
np.savetxt(outdir + "Jvals_individual.txt", obj.Jvals_individual)

if True:
    taylor_test(obj, xmin)
