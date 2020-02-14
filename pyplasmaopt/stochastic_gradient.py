import numpy as np


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
