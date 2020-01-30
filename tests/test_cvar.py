import pytest
import numpy as np
from pyplasmaopt import CVaR
np.random.seed(1)
import matplotlib
matplotlib.use('AGG')

def test_cvar_toy_example():
    nsamples = 1000
    perturbs = 1 * np.random.normal(size=(nsamples, 1))
    x = np.zeros((1,))
    t = np.zeros((1,))

    delta = 0.1
    def foo(x):
        res = 0.1+np.maximum(x, 0) * x**3 + np.minimum(x, 0) * delta * x
        return res.reshape((res.size,))

    def dfoo(x):
        return np.maximum(x, 0) * 4 * x**2 + np.minimum(x, 0) * delta * 2

    xmin_det = np.asarray([0])

    class Obj():
        def __init__(self):
            self.x = np.zeros((1,))

        def update(self, x):
            self.x = x
            x = perturbs + x
            self.J_samples = foo(x)
            self.dJ_samples = dfoo(x)

    alpha = 0.9
    obj = Obj()
    cvar = CVaR(obj, alpha, eps=1)
    tx = np.concatenate((t,x))
    from scipy.optimize import minimize

    x0 = np.asarray([1.0, 0])
    cvar.update(x0)
    h = np.asarray([1., 1])
    dJ = np.sum(cvar.dJ() * h)
    err = 1e8
    for i in range(1, 10):
        eps = 0.5**i
        cvar.update(x0 + eps * h)
        Jp = cvar.J()
        cvar.update(x0 - eps * h)
        Jm = cvar.J()
        err_new = 0.5*(Jp-Jm)/eps-dJ
        assert err_new < 0.55**2 * err
        err = err_new
        print(0.5*(Jp-Jm)/eps-dJ, 0.5*(Jp-Jm)/eps, dJ)

    def minimize_cvar(t, x):
        def fun(tx):
            cvar.update(tx)
            res = cvar.J()
            dres = cvar.dJ()
            return res, dres
        res = minimize(fun, np.array([t, x]), jac=True, method="BFGS", tol=1e-10)
        x = res.x[1]
        t = res.x[0]
        xpert = perturbs + x
        fxpert = foo(xpert)
        true_cvar = np.mean([f for f in fxpert if f >= np.quantile(fxpert, alpha)])
        print(res)
        print("eps=%.2e:" % cvar.eps, res.fun, "vs", true_cvar)
        return t, x

    for i in range(7):
        t, x = minimize_cvar(t, x)
        cvar.eps *= 0.1
    xs = np.linspace(-1, 1, 100)
    xpert = perturbs + x
    fxpert = foo(xpert)


    import matplotlib.pyplot as plt
    plt.plot(xs, foo(xs), zorder=1)
    plt.scatter(xpert, fxpert, color='r', s=0.1, zorder=2)
    plt.scatter([x], [foo(np.array([x]))], color='g', s=10, zorder=3)
    plt.scatter([xmin_det], [foo(xmin_det)], color='y', s=10, zorder=3)
    plt.savefig("/tmp/tmp.png", dpi=600)
    plt.close()
    plt.hist(fxpert, bins=50, log=True)
    plt.xscale('log')
    plt.savefig("/tmp/tmp_hist.png", dpi=600)

def test_cvar_2d_example():
    dim = 2
    nsamples = 1000
    perturbs = 0.4 * np.random.normal(size=(nsamples, dim))
    t = np.zeros((1,))
    x = np.zeros((dim,))

    def foo(x):
        res = np.exp(np.sum(x, axis=1)) + 0.01 * (np.sum(x**2, axis=1))
        return res

    def dfoo(x):
        return np.exp(np.sum(x, axis=1))[:, None]  + 2 * 0.01 * x

    class Obj():

        def update(self, x):
            self.x = x.copy()
            x = perturbs + x
            self.J_samples = foo(x)
            self.dJ_samples = dfoo(x)

    alpha = 0.95
    obj = Obj()
    cvar = CVaR(obj, alpha, eps=1)
    tx = np.concatenate((t, x))
    from scipy.optimize import minimize

    x0 = np.asarray([0.0, 0.1, 0.1])
    cvar.update(x0)
    h = np.asarray([1., 1, 1.])
    dJ = np.sum(cvar.dJ() * h)
    err = 1e8
    for i in range(1, 8):
        eps = 0.5**i
        cvar.update(x0 + eps * h)
        Jp = cvar.J()
        cvar.update(x0 - eps * h)
        Jm = cvar.J()
        err_new = 0.5*(Jp-Jm)/eps-dJ
        assert err_new < 0.55**2 * err
        err = err_new
        print(0.5*(Jp-Jm)/eps-dJ, 0.5*(Jp-Jm)/eps, dJ)

    def minimize_cvar(t, x):
        def fun(tx):
            cvar.update(tx)
            res = cvar.J()
            dres = cvar.dJ()
            return res, dres
        res = minimize(fun, np.concatenate((t, x)), jac=True, method="BFGS", tol=1e-10)
        x = res.x[1:]
        t = res.x[0:1]
        xpert = perturbs + x
        fxpert = foo(xpert)
        true_cvar = np.mean([f for f in fxpert if f >= np.quantile(fxpert, alpha)])
        print(res)
        print("eps=%.2e:" % cvar.eps, res.fun, "vs", true_cvar)
        return t, x

    def minimize_mean(x):
        def fun(x):
            obj.update(x)
            res = np.mean(obj.J_samples)
            dres = np.mean(obj.dJ_samples, axis=0)
            return res, dres
        res = minimize(fun, x, jac=True, method="BFGS", tol=1e-10)
        print(res)
        return res.x

    for i in range(5):
        t, x = minimize_cvar(t, x)
        cvar.eps *= 0.1

    x_cvar = x
    x_mean = minimize_mean(np.zeros((dim, )))

    x_cvar_pert = perturbs + x_cvar
    fx_cvar_pert = foo(x_cvar_pert)
    x_mean_pert = perturbs + x_mean
    fx_mean_pert = foo(x_mean_pert)

    assert np.mean(fx_mean_pert) < np.mean(fx_cvar_pert)

    fx_cvar_true_cvar = np.mean([f for f in fx_cvar_pert if f >= np.quantile(fx_cvar_pert, alpha)])
    fx_mean_true_cvar = np.mean([f for f in fx_mean_pert if f >= np.quantile(fx_mean_pert, alpha)])
    assert np.mean(fx_cvar_true_cvar) < np.mean(fx_mean_true_cvar)

    import matplotlib.pyplot as plt
    plt.hist(fx_cvar_pert, bins=100, log=True, alpha=0.75, label="Minimise CVaR")
    plt.hist(fx_mean_pert, bins=100, log=True, alpha=0.75, label="Minimise Mean")
    plt.legend()
    plt.xscale('log')
    plt.savefig("/tmp/tmp_hist.png", dpi=600)
