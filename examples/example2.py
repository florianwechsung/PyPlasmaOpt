from pyplasmaopt import *
from problem2_objective import Problem2_Objective
import numpy as np
import matplotlib.pyplot as plt

solver = "scipy"
# solver = "nlopt"
# solver = "pylbfgs"

nfp = 2
(coils, ma) = get_matt_data(nfp=nfp, ppp=20, at_optimum=True)
currents = [1e5 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
# currents = [0 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
stellarator = CoilCollection(coils, currents, nfp, True)
obj = Problem2_Objective(stellarator, ma, curvature_scale=1e-8, torsion_scale=1e-8, tikhonov=1e-4, eta_bar=-2.105800979374183)

# nfp = 3
# (coils, ma) = get_flat_data(nfp=nfp, ppp=20)
# currents = 3 * [0]
# stellarator = CoilCollection(coils, currents, nfp, True)
# obj = Problem2_Objective(stellarator, ma, curvature_scale=0e-8, torsion_scale=0e-8, tikhonov=0e-4, eta_bar=-2.25,
#                          iota_target=0.45, coil_length_target=2.19911485751, magnetic_axis_length_target=6.28318530718)



print(obj.x0.shape)
# obj.update(obj.x0.copy().reshape((175,1)))
info_dict = {'Nfeval':0}

def plot(info):
    if info['Nfeval'] % 10 == 0:
        ax = None
        for i in range(0, len(stellarator.coils)):
            ax = stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(coils)])
        ma.plot(ax=ax, show=False, closed_loop=False)
        ax.view_init(elev=90., azim=0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 1)
        plt.savefig('output/%i.png' % info['Nfeval'])
        plt.close()
        print("################################################################################")
        print(f"f eval {info['Nfeval']}")
        obj.print_status()
    info['Nfeval'] += 1

x = obj.x0
obj.update(x)
print("Initial value of J", obj.res)
if False:
    obj.update(x)
    J0, dJ0 = obj.res, obj.dres
    np.random.seed(1)
    h = 1e-3 * np.random.rand(*(x.shape))
    h[10:] = 0
    dJh = sum(dJ0*h)
    for i in range(5, 20):
        eps = 0.5**i
        obj.update(x + eps * h)
        Jeps = obj.res
        err = abs((Jeps-J0)/eps - dJh)
        print(err)
    import sys
    sys.exit()

maxiter = 100
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
    opt.set_vector_storage(50)
    opt.set_maxeval(maxiter)
    x = opt.optimize(list(x))
elif solver == "scipy":
    def J_scipy(x, info=None):
        obj.update(x)
        res = obj.res
        dres = obj.dres
        # if False:
        if info is not None:
            plot(info)
        return res, dres
    import time
    from scipy.optimize import minimize
    t1 = time.time()
    res = minimize(J_scipy, x, args=(info_dict,), jac=True, method='BFGS', tol=1e-20, options={'maxiter': maxiter})
    t2 = time.time()
    print(f"Time per iteration: {(t2-t1)/info_dict['Nfeval']:.4f}")
    print(res)
    x = res.x
    print("Gradient norm at minimum:", np.linalg.norm(res.jac))
    import sys
    sys.exit()
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

if False:
    obj.update(x)
    J0, dJ0 = obj.res, obj.dres
    np.random.seed(1)
    h = 1e-3 * np.random.rand(*(x.shape))
    h[10:] = 0
    dJh = sum(dJ0*h)
    for i in range(5, 20):
        eps = 0.5**i
        obj.update(x + eps * h)
        Jeps = obj.res
        err = abs((Jeps-J0)/eps - dJh)
        print(err)

sigma = 1e-4
sampler = GaussianSampler(coils[0].points, length_scale=0.2, sigma=sigma)
perturbed_coils = [GaussianPerturbedCurve(coil, sampler) for coil in stellarator.coils]
perturbed_bs = BiotSavart(perturbed_coils, stellarator.currents)
J_BSvsQS = BiotSavartQuasiSymmetricFieldDifference(obj.qsf, perturbed_bs)
J_BSvsQS.update()
# ax = None
# for i in range(0, len(stellarator.coils)):
#     ax = stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(coils)], linestyle="-")
# for i in range(0, len(perturbed_coils)):
#     ax = perturbed_coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(coils)], linestyle=":")
# ma.plot(ax=ax, show=False, closed_loop=False)
# ax.view_init(elev=90., azim=0)
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-1, 1)
# plt.show()

opt_L2 = obj.J_BSvsQS.J_L2()
opt_H1 = obj.J_BSvsQS.J_H1()
print(opt_L2, opt_H1)
L2s = []
H1s = []
nsamples = 2000
for i in range(nsamples):
    [coil.resample() for coil in perturbed_coils]
    J_BSvsQS.update_biotsavart()
    L2s.append(J_BSvsQS.J_L2())
    H1s.append(J_BSvsQS.J_H1())

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

bins = 100
plt.subplot(121)
plt.title("$\\frac{1}{2}||B_{BS}-B_{QS}||^2$")
plt.hist(L2s, bins, density=True, facecolor='g', alpha=0.75, log=True)
ymax = plt.ylim()[1]
plt.vlines(opt_L2, 0, ymax, color='r')
plt.xlim((0.5 * opt_L2, 2 * max(L2s)))
plt.xscale('log')
plt.subplot(122)
plt.title("$\\frac{1}{2}||\\nabla B_{BS}-\\nabla B_{QS}||^2$")
plt.hist(H1s, bins, density=True, facecolor='b', alpha=0.75, log=True)
ymax = plt.ylim()[1]
plt.vlines(opt_H1, 0, ymax, color='r')
plt.xscale('log')
plt.xlim((0.5 * opt_H1, 2 * max(H1s)))
plt.suptitle("$\\sigma= %.4f$" % sigma)
plt.savefig("sigma= %.4f.png" % sigma)
plt.show()




import IPython; IPython.embed()
# import sys
# sys.exit()
