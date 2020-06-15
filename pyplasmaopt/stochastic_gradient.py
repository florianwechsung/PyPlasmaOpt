import numpy as np
from .logging import warning


def gradient_descent(J, x0, learning_rate, maxiter, callback=lambda x: None, P=None):
    xk = x0.copy()
    f0 = None
    for i in range(maxiter):
        f, df = J(xk)
        if f0 is None:
            f0 = f
        else:
            if f > 10 * f0:
                break
        if P is not None:
            df = P @ df
        callback(xk)
        xk -= learning_rate * df
    return xk

def ada_grad(J, x0, learning_rate, maxiter, callback=lambda x: None, fudge_factor=1e-6, P=None):
    xk = x0.copy()
    gti = np.zeros(x0.shape[0])
    f0 = None
    for i in range(maxiter):
        f, df = J(xk)
        if f0 is None:
            f0 = f
        else:
            if f > 10 * f0:
                break
        if P is not None:
            df = P @ df
        callback(xk)
        gti += df**2
        adjusted_df = df / (fudge_factor + np.sqrt(gti))
        xk -= learning_rate * adjusted_df
    return xk

def momentum(J, x0, learning_rate, maxiter, callback=lambda x: None, momentum=0.9, P=None):
    xk = x0.copy()
    v = np.zeros(x0.shape[0])
    f0 = None
    for i in range(maxiter):
        f, df = J(xk)
        if f0 is None:
            f0 = f
        else:
            if f > 10 * f0:
                break
        if P is not None:
            df = P @ df
        callback(xk)
        v[:] = momentum * v + learning_rate * df
        xk -= v
    return xk

def rmsprop(J, x0, learning_rate, maxiter, callback=lambda x: None, fudge_factor=1e-6, P=None):
    eta = 0.1
    beta = 0.9
    eps = 1e-12
    xk = x0.copy()
    v = np.zeros(x0.shape[0])
    f0 = None
    for i in range(maxiter):
        f, df = J(xk)
        if f0 is None:
            f0 = f
        else:
            if f > 10 * f0:
                break
        if P is not None:
            df = P @ df
        callback(xk)
        v[:] = beta * v + (1-beta)*df**2
        xk -= (learning_rate/np.sqrt(v + eps)) * df
    return xk

def online_bfgs(J, x0, maxiter, callback=lambda x: None, B0=None, c=1.0, lam=1e-5, lr=0.1, tau=100):
    """
    Based on 'A Stochastic Quasi-Newton Method for Online Convex Optimization'
    by Schraudolph, Yu, and Günter.
    """
    xk = x0.copy()
    n = x0.shape[0]
    eps = 1e-10
    I = np.identity(n)
    if B0 is None:
        Bk = eps * I
    else:
        Bk = B0.copy()
    f0 = None
    for k in range(maxiter):
        etak = lr * tau/(tau+k)
        f, df = J(xk, resample=True)
        if f0 is None:
            f0 = f
        else:
            if f > 1000 * f0:
                break
        callback(xk)
        pk = - Bk@df
        sk = (etak/c) * pk
        xk += sk
        _, dfnew = J(xk, resample=False)
        yk = dfnew - df + lam*sk
        if k == 0 and B0 is None:
            Bk = (np.sum(sk*yk)/np.sum(yk*yk)) * I
        rhok = np.sum(sk*yk)**(-1)
        Bk = (I - rhok * np.outer(sk, yk)) @ Bk @ (I - rhok*np.outer(yk, sk)) + c*rhok*np.outer(sk, sk)
    return xk

def hybrid_bfgs(J, x0, maxiter, callback=lambda x: None, B0=None, c=1.0, lam=1e-5, lr=0.1, tau=100):
    """
    Based on 'A Stochastic Quasi-Newton Method for Online Convex Optimization'
    by Schraudolph, Yu, and Günter.
    """
    xk = x0.copy()
    n = x0.shape[0]
    eps = 1e-10
    I = np.identity(n)
    if B0 is None:
        Bk = eps * I
    else:
        Bk = B0.copy()
    f, df, df_det_old = J(xk, resample=True)
    f0 = None
    for k in range(maxiter):
        etak = lr * tau/(tau+k)
        callback(xk)
        if f0 is None:
            f0 = f
        else:
            if f > 1000 * f0:
                warning(f"Uah! {f} > 1000 * {f0}")
                break
        pk = - Bk@df
        sk = (etak/c) * pk
        xk += sk
        f, df, df_det = J(xk, resample=True)
        yk = df_det - df_det_old + lam*sk
        if k == 0 and B0 is None:
            Bk = (np.sum(sk*yk)/np.sum(yk*yk)) * I
        rhok = np.sum(sk*yk)**(-1)
        Bk = (I - rhok * np.outer(sk, yk)) @ Bk @ (I - rhok*np.outer(yk, sk)) + c*rhok*np.outer(sk, sk)
        df_det_old = df_det
    return xk
