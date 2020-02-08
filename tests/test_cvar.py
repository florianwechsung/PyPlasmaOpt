import pytest
import numpy as np
from pyplasmaopt import CVaR
np.random.seed(1)
import matplotlib
matplotlib.use('AGG')

def test_cvar_toy_example():
    nsamples = 100
    perturbs = 10 * np.random.normal(size=(nsamples, 1))**2
    x = np.zeros((1,))
    t = np.zeros((1,))

    delta = 0.1
    def foo(x):
        res = 0.1+np.maximum(x, 0) * x**3 + np.minimum(x, 0) * delta * x
        return res.reshape((res.size,))

    def dfoo(x):
        return np.maximum(x, 0) * 4 * x**2 + np.minimum(x, 0) * delta * 2

    xmin_det = np.asarray([0])

    alpha = 0.9
    cvar = CVaR(alpha, eps=1)

    class Obj():
        def J(self, tx):
            t = tx[0]
            x = perturbs + tx[1:]
            return cvar.J(t, foo(x))

        def dJ(self, tx):
            t = tx[0]
            x = perturbs + tx[1:]
            return np.concatenate((cvar.dJ_dt(t, foo(x)), cvar.dJ_dx(t, foo(x), dfoo(x))))

    obj = Obj()
    tx = np.concatenate((t,x))
    from scipy.optimize import minimize

    x0 = np.asarray([1.0, 0])
    h = np.asarray([1., 1])
    dJ = np.sum(obj.dJ(x0) * h)
    err = 1e8
    for i in range(1, 10):
        eps = 0.5**i
        Jp = obj.J(x0 + eps * h)
        Jm = obj.J(x0 - eps * h)
        err_new = 0.5*(Jp-Jm)/eps-dJ
        assert err_new < 0.55**2 * err
        err = err_new
        print(0.5*(Jp-Jm)/eps-dJ, 0.5*(Jp-Jm)/eps, dJ)

    def minimize_cvar(t, x):
        def fun(tx):
            res = obj.J(tx)
            dres = obj.dJ(tx)
            return res, dres
        res = minimize(fun, np.array([t, x]), jac=True, method="BFGS", tol=1e-10)
        x = res.x[1]
        t = res.x[0]
        xpert = perturbs + x
        fxpert = foo(xpert)
        true_cvar = np.mean([f for f in fxpert if f >= np.quantile(fxpert, alpha)])
        print(res)
        print("eps=%.2e:" % cvar.eps, res.fun, "vs", true_cvar, "err", abs(res.fun-true_cvar))
        return t, x

    for i in range(17):
        t, x = minimize_cvar(t, x)
        cvar.eps *= 0.2
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
        def J(self, tx):
            t = tx[0]
            x = perturbs + tx[1:]
            return cvar.J(t, foo(x))

        def dJ(self, tx):
            t = tx[0]
            x = perturbs + tx[1:]
            return np.concatenate((cvar.dJ_dt(t, foo(x)), cvar.dJ_dx(t, foo(x), dfoo(x))))

    alpha = 0.95
    obj = Obj()
    cvar = CVaR(alpha, eps=1)
    tx = np.concatenate((t, x))
    from scipy.optimize import minimize

    x0 = np.asarray([0.0, 0.1, 0.1])
    h = np.asarray([1., 1, 1.])
    dJ = np.sum(obj.dJ(x0) * h)
    err = 1e8
    for i in range(1, 8):
        eps = 0.5**i
        Jp = obj.J(x0 + eps * h)
        Jm = obj.J(x0 - eps * h)
        err_new = 0.5*(Jp-Jm)/eps-dJ
        assert err_new < 0.55**2 * err
        err = err_new
        print(0.5*(Jp-Jm)/eps-dJ, 0.5*(Jp-Jm)/eps, dJ)

    def minimize_cvar(t, x):
        def fun(tx):
            res = obj.J(tx)
            dres = obj.dJ(tx)
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
            res = np.mean(foo(x + perturbs))
            dres = np.mean(dfoo(x + perturbs), axis=0)
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

if __name__ == "__main__":
    test_cvar_toy_example()
