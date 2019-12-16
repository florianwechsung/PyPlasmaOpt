from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve, \
    BiotSavartQuasiSymmetricFieldDifference, get_matt_data, CoilCollection, \
    CurveLength, CurveCurvature, QuasiSymmetricField, CurveTorsion
from problem2_objective import Problem2_Objective
import numpy as np
import matplotlib.pyplot as plt

nfp = 2
(coils, ma) = get_matt_data(nfp=nfp, ppp=20, at_optimum=False)
currents = [1e5 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
stellerator = CoilCollection(coils, currents, nfp, True)


obj = Problem2_Objective(stellerator, ma, curvature_scale=0, torsion_scale=0, tikhonov=1e-2)

def J(x, info=None):

    obj.update(x)
    res = obj.res
    dres = obj.dres
    # if False:
    if info is not None:
        if info['Nfeval'] % 10 == 0:
            ax = None
            for i in range(0, len(stellerator.coils)):
                ax = stellerator.coils[i].plot(ax=ax, show=False)
            ma.plot(ax=ax, show=False, closed_loop=False)
            ax.view_init(elev=90., azim=0)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1, 1)
            plt.savefig('output/%i.png' % info['Nfeval'])
            plt.close()
            print("################################################################################")
            print(f"Iteration {info['Nfeval']}")
            obj.print_status()
        info['Nfeval'] += 1
    return res, dres

x = obj.x0
if False:
    J0, dJ0 = J(x)
    np.random.seed(1)
    h = 1e-3 * np.random.rand(*(x.shape))
    h[10:] = 0
    dJh = sum(dJ0*h)
    for i in range(5, 20):
        eps = 0.5**i
        Jeps = J(x + eps * h)[0]
        err = abs((Jeps-J0)/eps - dJh)
        print(err)
    import sys
    sys.exit()

import time
from scipy.optimize import minimize
maxiter = 1000
t1 = time.time()
args = {'Nfeval':0}
res = minimize(J, x, args=(args,), jac=True, method='BFGS', tol=1e-20, options={'maxiter': maxiter})
t2 = time.time()
print(f"Time per iteration: {(t2-t1)/args['Nfeval']:.4f}")
print(res)
x = res.x
print("Gradient norm at minimum:", np.linalg.norm(res.jac), np.linalg.norm(J(x)[1]))

# J0, dJ0 = J(coil_dofs)
# h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
# dJh = sum(dJ0*h)
# for i in range(5, 10):
#     eps = 0.5**i
#     Jeps = J(coil_dofs + eps * h)[0]
#     err = abs((Jeps-J0)/eps - dJh)
#     print(err)
# import IPython; IPython.embed()
ax = None
for i in range(0, len(stellerator.coils)):
    ax = stellerator.coils[i].plot(ax=ax, show=False)
ma.plot(ax=ax, show=False, closed_loop=False)
ax.view_init(elev=90., azim=0)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
plt.show()
# import sys
# sys.exit()
