from cached_property import cached_property
from sympy import *
import numpy as np
from math import pi, sin, cos

class Curve():
    r"""
    A periodic curve \Gamma : [0, 1) \to R^3, \phi\mapsto\Gamma(\phi).
    """

    def __init__(self, points):
        self.points = points
        self.dependencies = []

    def clear_cache(self):
        for obj in [self] + self.dependencies:
            for prop in ["gamma", "dgamma_by_dphi", "d2gamma_by_dphidphi",
                         "d2gamma_by_dphidphi", "kappa", "dgamma_by_dcoeff",
                         "d2gamma_by_dphidcoeff", "d3gamma_by_dphidphidcoeff",
                         "dkappa_by_dcoeff", "incremental_arclength",
                         "dincremental_arclength_by_dcoeff",
                         "dincremental_arclength_by_dphi", "frenet_frame"]:
                if prop in obj.__dict__:
                    del obj.__dict__[prop]


    def gamma_impl(self, points):
        """ Evaluate the curve at `points`. """
        raise NotImplementedError

    @cached_property
    def gamma(self):
        return self.gamma_impl(self.points)


    def dgamma_by_dphi_impl(self, points):
        """ Return the derivative of the curve. """
        raise NotImplementedError

    @cached_property
    def dgamma_by_dphi(self):
        return self.dgamma_by_dphi_impl(self.points)


    def d2gamma_by_dphidphi_impl(self):
        """ Return the second derivative of the curve. """
        raise NotImplementedError

    @cached_property
    def d2gamma_by_dphidphi(self):
        return self.d2gamma_by_dphidphi_impl(self.points)


    def kappa_impl(self, points):
        """ Curvature at `points`. """
        dgamma = self.dgamma_by_dphi_impl(points)[:, 0, :]
        d2gamma = self.d2gamma_by_dphidphi_impl(points)[:, 0, 0, :]
        return (np.linalg.norm(np.cross(dgamma, d2gamma), axis=1)/np.linalg.norm(dgamma, axis=1)**3).reshape(len(points),1)

    @cached_property
    def kappa(self):
        return self.kappa_impl(self.points)


    def dgamma_by_dcoeff_impl(self, points):
        raise NotImplementedError

    @cached_property
    def dgamma_by_dcoeff(self):
        return self.dgamma_by_dcoeff_impl(self.points)


    def d2gamma_by_dphidcoeff_impl(self, points):
        raise NotImplementedError

    @cached_property
    def d2gamma_by_dphidcoeff(self):
        return self.d2gamma_by_dphidcoeff_impl(self.points)


    def d3gamma_by_dphidphidcoeff_impl(self, points):
        raise NotImplementedError

    @cached_property
    def d3gamma_by_dphidphidcoeff(self):
        return self.d3gamma_by_dphidphidcoeff_impl(self.points)


    def dkappa_by_dcoeff_impl(self, points):
        dgamma_by_dphi = self.dgamma_by_dphi_impl(points)[:, 0, :]
        dgamma_by_dphidphi = self.d2gamma_by_dphidphi_impl(points)[:, 0, 0, :]
        dgamma_by_dphidcoeff = self.d2gamma_by_dphidcoeff_impl(points)[:, 0, :, :]
        dgamma_by_dphidphidcoeff = self.d3gamma_by_dphidphidcoeff_impl(points)[:, 0, 0, :, :]
        num_coeff = dgamma_by_dphidcoeff.shape[1]
        res = np.zeros((len(points), num_coeff, 1))
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        numerator = np.cross(dgamma_by_dphi, dgamma_by_dphidphi)
        denominator = norm(dgamma_by_dphi)
        for i in range(num_coeff):
            res[:, i, 0] = (
                +inner(
                    numerator, 
                    np.cross(dgamma_by_dphidcoeff[:, i,:], dgamma_by_dphidphi) + np.cross(dgamma_by_dphi, dgamma_by_dphidphidcoeff[:, i, :])
                ) * denominator**3 / norm(numerator)
                - norm(numerator) * 3 * denominator * inner(dgamma_by_dphi, dgamma_by_dphidcoeff[:, i, :])
            )/denominator**6
        return res

    @cached_property
    def dkappa_by_dcoeff(self):
        return self.dkappa_by_dcoeff_impl(self.points)


    def incremental_arclength_impl(self, points):
        return np.linalg.norm(self.dgamma_by_dphi_impl(points)[:, 0, :], axis=1).reshape((len(points), 1))

    @cached_property
    def incremental_arclength(self):
        return self.incremental_arclength_impl(self.points)


    def dincremental_arclength_by_dcoeff_impl(self, points):
        dgamma_by_dphi = self.dgamma_by_dphi_impl(points)[:, 0, :]
        dgamma_by_dphidcoeff = self.d2gamma_by_dphidcoeff_impl(points)[:, 0, :, :]
        num_coeff = dgamma_by_dphidcoeff.shape[1]
        res = np.zeros((len(points), num_coeff, 1))
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        for i in range(num_coeff):
            res[:, i, 0] = inner(dgamma_by_dphi, dgamma_by_dphidcoeff[:, i, :])/norm(dgamma_by_dphi)
        return res

    @cached_property
    def dincremental_arclength_by_dcoeff(self):
        return self.dincremental_arclength_by_dcoeff_impl(self.points)


    def dincremental_arclength_by_dphi_impl(self, points):
        dgamma_by_dphi = self.dgamma_by_dphi_impl(points)[:, 0, :]
        dgamma_by_dphidphi = self.d2gamma_by_dphidphi_impl(points)[:, 0, 0, :]
        res = np.zeros((len(points), 1, 1))
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        res[:, 0, 0] = inner(dgamma_by_dphi, dgamma_by_dphidphi)/norm(dgamma_by_dphi)
        return res

    @cached_property
    def dincremental_arclength_by_dphi(self):
        return self.dincremental_arclength_by_dphi_impl(self.points)


    def frenet_frame_impl(self, points):
        """
        Returns the (t, n, b) Frenet frame.
        """
        dgamma_by_dphi = self.dgamma_by_dphi_impl(points)[:, 0, :]
        d2gamma_by_dphidphi = self.d2gamma_by_dphidphi_impl(points)[:, 0, 0, :]
        l = self.incremental_arclength_impl(points)
        dl_by_dphi = self.dincremental_arclength_by_dphi_impl(points)[:, 0, 0]
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)

        t = (1./norm(dgamma_by_dphi))[:, None] * dgamma_by_dphi

        tdash = (1/norm(dgamma_by_dphi)**2)[:, None] * (
            norm(dgamma_by_dphi)[:, None] * d2gamma_by_dphidphi
            - (inner(dgamma_by_dphi, d2gamma_by_dphidphi)/norm(dgamma_by_dphi))[:, None] *  dgamma_by_dphi
        )
        kappa = self.kappa_impl(points)
        n = (1./norm(tdash))[:, None] * tdash
        b = np.cross(t, n, axis=1)
        return (t, n, b)

    @cached_property
    def frenet_frame(self):
        return self.frenet_frame_impl(self.points)


    def plot(self, resolution=None, ax=None, show=True, plot_derivative=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if resolution is None:
            gamma = self.gamma
            dgamma_by_dphi = self.dgamma_by_dphi
        else:
            phis = np.linspace(0, 1, resolution)
            gamma = np.stack(self.gamma_impl(phis))
            dgamma_by_dphi = self.dgamma_by_dphi_impl(phis)[:, 0, :]
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2])
        if plot_derivative:
            ax.quiver(gamma[:, 0], gamma[:, 1], gamma[:, 2], 0.1 * dgamma_by_dphi[:, 0], 0.1 * dgamma_by_dphi[:, 1], 0.1 * dgamma_by_dphi[:, 2], arrow_length_ratio=0.1, color="r")
        if show:
            plt.show()
        return ax


class CartesianFourierCurve(Curve):
    r"""
A curve of the form 
    \Gamma(\phi) = e_1 * x_1(\phi) + e_2 * x_2(\phi) + e_3 * x_3(\phi)
with
    x_i(\phi) = coeff[i][0] + \sum_{j=1}^terms coeff[i][2*j-1] * sin(2*pi*j*\phi) + coeff[i][2*j] * cos(2*pi*j*\phi)
    """

    def __init__(self, order, *args):
        super().__init__(*args)
        self.coefficients = [np.zeros((2*order+1,)), np.zeros((2*order+1,)), np.zeros((2*order+1,))]
        self.order = order

    def num_coeff(self):
        return 3*(2*self.order+1)

    def get_dofs(self):
        return np.concatenate(self.coefficients)

    def set_dofs(self, dofs):
        self.clear_cache()
        counter = 0
        for i in range(3):
            self.coefficients[i][0] = dofs[counter]
            counter += 1
            for j in range(1, self.order+1):
                self.coefficients[i][2*j-1] = dofs[counter]
                counter += 1
                self.coefficients[i][2*j] = dofs[counter]
                counter += 1

    def gamma_impl(self, points):
        res = np.zeros((len(points), 3))
        coeffs = self.coefficients
        for i in range(3):
            res[:, i] += coeffs[i][0]
            for j in range(1, self.order+1):
                res[:, i] += coeffs[i][2*j-1] * np.sin(2*pi*j*points)
                res[:, i] += coeffs[i][2*j]   * np.cos(2*pi*j*points)
        return res

    def dgamma_by_dcoeff_impl(self, points):
        res = np.zeros((len(points), self.num_coeff(), 3))
        for i in range(3):
            res[:, i*(2*self.order+1), i] = 1
            for j in range(1, self.order+1):
                res[:, i*(2*self.order+1) + 2*j-1, i] = np.sin(2*pi*j*points)
                res[:, i*(2*self.order+1) + 2*j  , i] = np.cos(2*pi*j*points)
        return res

    def dgamma_by_dphi_impl(self, points):
        res = np.zeros((len(points), 1, 3))
        coeffs = self.coefficients
        for i in range(3):
            for j in range(1, self.order+1):
                res[:, 0, i] += +coeffs[i][2*j-1] * 2*pi*j*np.cos(2*pi*j*points)
                res[:, 0, i] += -coeffs[i][2*j] * 2*pi*j*np.sin(2*pi*j*points)
        return res

    def d2gamma_by_dphidcoeff_impl(self, points):
        res = np.zeros((len(points), 1, self.num_coeff(), 3))
        for i in range(3):
            for j in range(1, self.order+1):
                res[:, 0, i*(2*self.order+1) + 2*j-1, i] = +2*pi*j*np.cos(2*pi*j*points)
                res[:, 0, i*(2*self.order+1) + 2*j  , i] = -2*pi*j*np.sin(2*pi*j*points)
        return res

    def d2gamma_by_dphidphi_impl(self, points):
        res = np.zeros((len(points), 1, 1, 3))
        coeffs = self.coefficients
        for i in range(3):
            for j in range(1, self.order+1):
                res[:, 0, 0, i] += -coeffs[i][2*j-1] * (2*pi*j)**2*np.sin(2*pi*j*points)
                res[:, 0, 0, i] += -coeffs[i][2*j]   * (2*pi*j)**2*np.cos(2*pi*j*points)
        return res

    def d3gamma_by_dphidphidcoeff_impl(self, points):
        res = np.zeros((len(points), 1, 1, self.num_coeff(), 3))
        for i in range(3):
            for j in range(1, self.order+1):
                res[:, 0, 0, i*(2*self.order+1) + 2*j-1, i] = -(2*pi*j)**2*np.sin(2*pi*j*points)
                res[:, 0, 0, i*(2*self.order+1) + 2*j  , i] = -(2*pi*j)**2*np.cos(2*pi*j*points)
        return res



class StelleratorSymmetricCylindricalFourierCurve(Curve):

    def __init__(self, order, nfp, *args):
        super().__init__(*args)
        self.coefficients = [np.zeros((order+1,)), np.zeros((order,))]
        self.nfp = nfp
        self.order = order

    def num_coeff(self):
        return 2*self.order+1

    def get_dofs(self):
        return np.concatenate(self.coefficients)

    def set_dofs(self, dofs):
        self.clear_cache()
        counter = 0
        for i in range(self.order+1):
            self.coefficients[0][i] = dofs[i]
        for i in range(self.order):
            self.coefficients[1][i] = dofs[self.order + 1 + i]

    def gamma_impl(self, points):
        res = np.zeros((len(points), 3))
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, 0] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            res[:, 1] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            res[:, 2] += self.coefficients[1][i-1] * np.sin(nfp * 2 * pi * i * points)
        return res

    def dgamma_by_dcoeff_impl(self, points):
        res = np.zeros((len(points), self.num_coeff(), 3))
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, i, 0] = np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            res[:, i, 1] = np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            res[:, self.order + i, 2] = np.sin(nfp * 2 * pi * i * points)
        return res

    def dgamma_by_dphi_impl(self, points):
        res = np.zeros((len(points), 1, 3))
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, 0, 0] += self.coefficients[0][i] * (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
            res[:, 0, 1] += self.coefficients[0][i] * (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
        for i in range(1, self.order+1):
            res[:, 0, 2] += self.coefficients[1][i-1] * (nfp * 2 * pi * i) * np.cos(nfp * 2 * pi * i * points)
        return res

    def d2gamma_by_dphidcoeff_impl(self, points):
        res = np.zeros((len(points), 1, self.num_coeff(), 3))
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, 0, i, 0] = (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
            res[:, 0, i, 1] = (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
        for i in range(1, self.order+1):
            res[:, 0, self.order + i, 2] = (nfp * 2 * pi * i) * np.cos(nfp * 2 * pi * i * points)
        return res

    def d2gamma_by_dphidphi_impl(self, points):
        res = np.zeros((len(points), 1, 1, 3))
        coeffs = self.coefficients
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, 0, 0, 0] += self.coefficients[0][i] * (
                -(nfp * 2 * pi * i)**2 *       np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +(nfp * 2 * pi * i)*(2 * pi) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +(nfp * 2 * pi * i)*(2 * pi) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -(2 * pi)**2 *                 np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
            res[:, 0, 0, 1] += self.coefficients[0][i] * (
                -(nfp * 2 * pi * i)**2 *         np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -(nfp * 2 * pi * i) * (2 * pi) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi) * (nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi)**2 *                   np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
        for i in range(1, self.order+1):
            res[:, 0, 0, 2] -= self.coefficients[1][i-1] * (nfp * 2 * pi * i)**2 * np.sin(nfp * 2 * pi * i * points)
        return res

    def d3gamma_by_dphidphidcoeff_impl(self, points):
        res = np.zeros((len(points), 1, 1, self.num_coeff(), 3))
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, 0, 0, i, 0] = (
                -(nfp * 2 * pi * i)**2 *       np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +(nfp * 2 * pi * i)*(2 * pi) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +(nfp * 2 * pi * i)*(2 * pi) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -(2 * pi)**2 *                 np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
            res[:, 0, 0, i, 1] = (
                -(nfp * 2 * pi * i)**2 *         np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -(nfp * 2 * pi * i) * (2 * pi) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi) * (nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi)**2 *                   np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
        for i in range(1, self.order+1):
            res[:, 0, 0, self.order + i, 2] = -(nfp * 2 * pi * i)**2 * np.sin(nfp * 2 * pi * i * points)
        return res

class RotatedCurve(Curve):

    def __init__(self, curve, theta, flip):
        super().__init__(curve.points)
        self.rotmat = np.asarray([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ]).T
        if flip:
            self.rotmat = self.rotmat @ np.asarray([[1,0,0],[0,-1,0],[0,0,-1]])
        self.curve = curve
        curve.dependencies.append(self)

    def gamma_impl(self, points):
        gamma = self.curve.gamma_impl(points)
        return gamma @ self.rotmat

    def dgamma_by_dphi_impl(self, points):
        dgamma_by_dphi = self.curve.dgamma_by_dphi_impl(points)
        return dgamma_by_dphi @ self.rotmat

    def d2gamma_by_dphidphi_impl(self, points):
        d2gamma_by_dphidphi = self.curve.d2gamma_by_dphidphi_impl(points)
        return d2gamma_by_dphidphi @ self.rotmat

    def dgamma_by_dcoeff_impl(self, points):
        dgamma_by_dcoeff = self.curve.dgamma_by_dcoeff_impl(points)
        return dgamma_by_dcoeff @ self.rotmat

    def d2gamma_by_dphidcoeff_impl(self, points):
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff_impl(points)
        return d2gamma_by_dphidcoeff @ self.rotmat

    def d3gamma_by_dphidphidcoeff_impl(self, points):
        d3gamma_by_dphidphidcoeff = self.curve.d3gamma_by_dphidphidcoeff_impl(points)
        return d3gamma_by_dphidphidcoeff @ self.rotmat

class GaussianCurve(Curve):

    def __init__(self, points, sigma, length_scale):
        super().__init__(points)
        xs = self.points
        n = len(xs)
        n_derivs = 2
        cov_mat = np.zeros((n*(n_derivs+1), n*(n_derivs+1)))
        def kernel(x, y):
            return sigma**2*exp(-(x-y)**2/length_scale**2)
        for ii in range(n_derivs+1):
            for jj in range(n_derivs+1):
                if ii + jj == 0:
                    lam = kernel
                else:
                    x = Symbol("x")
                    y = Symbol("y")
                    f = kernel(x, y)
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])))
                for i in range(n):
                    for j in range(n):
                        x = xs[i]
                        y = xs[j]
                        if abs(x-y)>0.5:
                            if y>0.5:
                                y -= 1
                            else:
                                x-= 1
                        cov_mat[ii*n + i, jj*n + j] = lam(x, y)

        from scipy.linalg import sqrtm
        L = np.real(sqrtm(cov_mat))
        z = np.random.normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = L@z

        self.gamma = curve_and_derivs[0:n,:]
        self.dgamma_by_dphi = curve_and_derivs[n:2*n,:]
        self.d2gamma_by_dphidphi = curve_and_derivs[2*n:3*n,:]
