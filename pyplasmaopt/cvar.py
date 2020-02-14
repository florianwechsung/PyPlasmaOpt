import numpy as np

def soft_plus(x, eps, method=2):
    # We implement two different differentiable approximations to function x
    # \mapsto max(x, 0). In the first one occasionally yields NaN or Inf, so we
    # prefer the second variant.
    if method==1:
        f = eps*np.log(1+np.exp(x/eps))
    else:
        x = x+eps/2
        f = (x**3/eps**2 - x**4/(2*eps**3)) * (x>0) * (x<eps) \
            + (x-eps/2) * (x>=eps)
    return f

def soft_plus_dash(x, eps, method=2):
    if method==1:
        dfdx = np.exp(x/eps)/(np.exp(x/eps)+1)
    else:
        x = x+eps/2
        dfdx = (3*x**2/eps**2 - 2*x**3/(eps**3)) * (x>0) * (x<eps) \
            + 1. * (x>=eps)
    return dfdx

class CVaR():

    def __init__(self, alpha, eps=0.1):
        self.alpha = alpha
        self.eps = eps

    def J(self, t, Jsamples):
        return t + np.mean(soft_plus(Jsamples-t, self.eps))/(1-self.alpha)
    
    def dJ_dt(self, t, Jsamples):
        softplus_dash = soft_plus_dash(Jsamples-t, self.eps)
        partial_t = np.asarray([1. - np.mean(softplus_dash)/(1-self.alpha)])
        return partial_t

    def dJ_dx(self, t, Jsamples, dJsamples):
        softplus_dash = soft_plus_dash(Jsamples-t, self.eps)
        partial_x = np.mean(softplus_dash[:, None] * dJsamples, axis=0)/(1-self.alpha)
        return partial_x

    def find_optimal_t(self, Jsamples, tinit=None):
        if tinit is None:
            tinit = np.asarray([0.])
        else:
            tinit = np.asarray([tinit])
        from scipy.optimize import minimize
        def J(t):
            return self.J(t, Jsamples), self.dJ_dt(t, Jsamples)
        res = minimize(J, tinit, jac=True, method='BFGS', tol=1e-10, options={'maxiter': 100})
        t = res.x[0]
        return t

