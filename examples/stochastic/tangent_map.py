import numpy as np
import scipy
from pyplasmaopt import *

def compute_on_axis(bs, ma):
    ma_points = 300
    madata_pert = find_magnetic_axis(bs, ma_points, np.linalg.norm(ma.gamma()[0, 0:2]), output='cartesian')
    ma_pert = CurveRZFourier(ma_points, 15, 1, False)
    ma_pert.least_squares_fit(madata_pert)
    # B = bs.set_points(ma_pert.gamma()).B()
    # absB = np.linalg.norm(B, axis=1)
    # l = ma_pert.incremental_arclength()

    # qs_l2 = np.mean(absB)
    # qs_L2 = np.mean(absB*l)/np.mean(l)

    # non_qs_l2 = np.mean((absB-qs_l2)**2)**0.5
    # non_qs_L2 = np.mean((absB-qs_L2)**2*l)**0.5

    t = TangentMap(bs, ma_pert, rtol=1e-12, atol=1e-12,
               bvp_tol=1e-8, tol=1e-12,
               verbose=0, nphi_guess=100,
               maxiter=50, method='RK45')
    return t.compute_iota()[0]

class TangentMap():
    def __init__(self, biotsavart, magnetic_axis, rtol=1e-12, atol=1e-12,
                 bvp_tol=1e-8, tol=1e-12,
                 verbose=0, nphi_guess=100,
                 maxiter=50, method='RK45'):
        """
        magnetic_axis: instance of StelleratorSymmetricCylindricalFourierCurve
            representing magnetic axis
        rtol (double): relative tolerance for IVP
        atol (double): absolute tolerance for IVP
        bvp_tol (double): tolerance for BVP 
        tol (double): tolerance for Newton solve
        verbose (int): verbosity for BVP and Newton solves
        nphi_guess (int): number of grid points for guess of axis solutions
        maxiter (int): maximum number of Newton iterations for axis solve
        axis_bvp (bool): if True, scipy.integrate.solve_bvp is used to solve for 
            axis. If False, Newton method is used. 
        adjoint_axis_bvp (bool): if True, scipy.integrate.solve_bvp is used to 
            solve for adjoint axis. If False, Newton method is used.             
        method (str): algorithm to use for scipy.integrate.solve_ivp
        """

        self.biotsavart = biotsavart
        self.magnetic_axis = magnetic_axis
        self.rtol = rtol
        self.atol = atol
        self.tol = tol
        self.verbose = verbose
        self.nphi_guess = nphi_guess
        self.maxiter = maxiter
        self.method = method
        # Polynomial solutions for current solutions
        self.axis_poly = None
        self.tangent_poly = None

    def update_solutions(self):
        """
        Computes solutions for the magnetic axis, tangent map, and corresponding
            adjoint solutions

        Inputs:
            derivatives (bool): If True, adjoint solutions required for 
                derivatives are computed.
        """
        phi = np.linspace(0, 2*np.pi, self.nphi_guess, endpoint=True)
        phi_reverse = np.linspace(2*np.pi, 0, self.nphi_guess, endpoint=True)

        sol, self.tangent_poly = self.compute_tangent(phi)

    def reset_solutions(self):
        """
        Reset solutions
        """
        self.axis_poly = None
        self.tangent_poly = None

    def compute_iota(self):
        """
        Compute rotational transform from tangent map.
        Outputs:
            iota (double): value of rotational transform.
        """
        phi = np.array([2*np.pi])
        if self.tangent_poly is None:
            self.update_solutions()
        M = self.tangent_poly(phi)
        detM = M[0]*M[3] - M[1]*M[2]
        np.testing.assert_allclose(detM, 1, rtol=1e-2)
        trM = M[0] + M[3]
        if (np.abs(trM/2) > 1):
            raise RuntimeError('Incorrect value of trM.')
        else:
            return np.arccos(trM/2)/(2*np.pi)

    def compute_tangent(self, phi, axis_poly=None, adjoint=False):
        """
        Compute tangent map by solving initial value problem.
        Inputs:
            phi (1d array): 1d array for evaluation of tangent map
            axis_poly: polynomial solution for axis
            adjoint (bool): if True, Jacobian matrix for adjoint axis integration
                is computed
        Outputs:
            y (2d array (4,len(phi))): flattened tangent map on grid of
                toroidal angle
        """

        y0 = np.array([1, 0, 0, 1])
        t_span = (0, 2*np.pi)
        out = scipy.integrate.solve_ivp(self.rhs_fun, t_span, y0,
                                        vectorized=False, rtol=self.rtol, atol=self.atol,
                                        t_eval=phi, dense_output=True,
                                        method=self.method)
        if (out.status == 0):
            return out.y, out.sol
        else:
            raise RuntimeError('Error ocurred in integration of tangent map.')

    def compute_m(self, phi, axis=None):
        """
        Computes the matrix that appears on the rhs of the tangent map ODE,
            e.g. M'(phi) = m(phi), for given phi.
        Inputs:
            phi (double): toroidal angle for evaluation
            axis (2d array (2,npoints)): R and Z for current axis state
        Outputs:
            m (1d array (4)): matrix appearing on rhs of tangent map ODE
        """

        points = phi/(2*np.pi)
        if (np.ndim(points) == 0):
            points = np.array([points])
        magamma = np.zeros((points.size, 3))
        self.magnetic_axis.gamma_impl(magamma, points)

        self.biotsavart.set_points(magamma)
        X = magamma[..., 0]
        Y = magamma[..., 1]
        Z = magamma[..., 2]

        gradB = self.biotsavart.dB_by_dX()
        B = self.biotsavart.B()
        BX = B[..., 0]
        BY = B[..., 1]
        BZ = B[..., 2]
        dBXdX = gradB[..., 0, 0]
        dBXdY = gradB[..., 1, 0]
        dBXdZ = gradB[..., 2, 0]
        dBYdX = gradB[..., 0, 1]
        dBYdY = gradB[..., 1, 1]
        dBYdZ = gradB[..., 2, 1]
        dBZdX = gradB[..., 0, 2]
        dBZdY = gradB[..., 1, 2]
        dBZdZ = gradB[..., 2, 2]
        R = np.sqrt(X**2 + Y**2)
        BR = X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        dBRdR = (X**2*dBXdX + X*Y * (dBYdX + dBXdY) + Y**2 * dBYdY)/(R**2)
        dBPdR = (X*Y * (dBYdY-dBXdX) + X**2 * dBYdX - Y**2 * dBXdY)/(R**2)
        dBZdR = dBZdX*X/R + dBZdY*Y/R
        dBRdZ = X*dBXdZ/R + Y*dBYdZ/R
        dBPdZ = -Y*dBXdZ/R + X*dBYdZ/R
        if (np.ndim(phi) == 0):
            m = np.zeros((4, 1))
        else:
            m = np.zeros((4, len(phi)))
        m[0, ...] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
        m[1, ...] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
        m[2, ...] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
        m[3, ...] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
        return np.squeeze(m)

    def rhs_fun(self, phi, M):
        """
        Computes the RHS of the tangent map ode, e.g. M'(phi) = rhs
            for given phi and M
        Inputs:
            phi (double): toroidal angle for evaluation
            M (1d array (4)): current value of tangent map
        Outputs:
            rhs (1d array (4)): rhs of ODE
        """
        m = self.compute_m(phi)
        assert(np.ndim(phi) == 0)

        out = np.squeeze(np.array([m[0]*M[0] + m[1]*M[2], m[0]*M[1] + m[1]*M[3],
                                   m[2]*M[0] + m[3]*M[2], m[2]*M[1] + m[3]*M[3]]))
        return out
