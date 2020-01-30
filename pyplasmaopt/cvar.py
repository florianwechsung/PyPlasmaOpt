import numpy as np

def soft_plus(x, eps, method=2):
    # We implement two different differentiable approximations to function x
    # \mapsto max(x, 0). In the first one occasionally yields NaN or Inf, so we
    # prefer the second variant.
    if method==1:
        f = eps*np.log(1+np.exp(x/eps))
    else:
        f = (x**3/eps**2 - x**4/(2*eps**3)) * (x>0) * (x<eps) \
            + (x-eps/2) * (x>=eps)
    return f

def soft_plus_dash(x, eps, method=2):
    if method==1:
        dfdx = np.exp(x/eps)/(np.exp(x/eps)+1)
    else:
        dfdx = (3*x**2/eps**2 - 2*x**3/(eps**3)) * (x>0) * (x<eps) \
            + 1. * (x>=eps)
    return dfdx

class CVaR():

    def __init__(self, obj, alpha, eps=0.1):
        self.obj = obj
        self.alpha = alpha
        self.eps = eps

    def update(self, tx):
        self.tx = tx.copy()
        self.obj.update(self.tx[1:])

    def J(self):
        t = self.tx[0]
        x = self.tx[1:]
        vals = self.obj.J_samples
        return t + np.mean(soft_plus(vals-t, self.eps))/(1-self.alpha)
    
    def dJ(self):
        t = self.tx[0]
        x = self.tx[1:]
        vals = self.obj.J_samples
        softplus_dash = soft_plus_dash(vals-t, self.eps)
        partial_t = np.asarray([1. - np.mean(softplus_dash)/(1-self.alpha)])
        partial_x = np.mean(softplus_dash[:, None] * self.obj.dJ_samples, axis=0)/(1-self.alpha)
        return np.concatenate((partial_t, partial_x))
