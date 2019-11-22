import numpy as np
from math import pi, sin, cos

class Curve():
    r"""
    A periodic curve \Gamma : [0, 1) \to R^3, \phi\mapsto\Gamma(\phi).
    """

    def gamma(self, points):
        """ Evaluate the curve at `points`. """
        raise NotImplementedError

    def dgamma_by_dphi(self, points):
        """ Return the derivative of the curve. """
        raise NotImplementedError

    def d2gamma_by_dphidphi(self, points):
        """ Return the second derivative of the curve. """
        raise NotImplementedError

    def kappa(self, points):
        """ Curvature at `points`. """
        dgamma = self.dgamma_by_dphi(points)[:, 0, :]
        d2gamma = self.d2gamma_by_dphidphi(points)[:, 0, 0, :]
        return np.linalg.norm(np.cross(dgamma, d2gamma), axis=1)/np.linalg.norm(dgamma, axis=1)**3

    def dgamma_by_dcoeff(self, points):
        raise NotImplementedError

    def d2gamma_by_dphidcoeff(self, points):
        raise NotImplementedError

    def d3gamma_by_dphidphidcoeff(self, points):
        raise NotImplementedError

    def dkappa_by_dcoeff(self, points):
        dgamma_by_dphi = self.dgamma_by_dphi(points)[:, 0, :]
        dgamma_by_dphidphi = self.d2gamma_by_dphidphi(points)[:, 0, 0, :]
        dgamma_by_dphidcoeff = self.d2gamma_by_dphidcoeff(points)[:, 0, :, :]
        dgamma_by_dphidphidcoeff = self.d3gamma_by_dphidphidcoeff(points)[:, 0, 0, :, :]
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

    def plot(self, resolution=100, ax=None, show=True, plot_derivative=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        phis = np.linspace(0, 1, resolution)
        gammas = np.stack(self.gamma(phis))
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        ax.plot(gammas[:, 0], gammas[:, 1], gammas[:, 2])
        if plot_derivative:
            dgamma_by_dphi = self.dgamma_by_dphi(phis)[:, 0, :]
            ax.quiver(gammas[:, 0], gammas[:, 1], gammas[:, 2], 0.1 * dgamma_by_dphi[:, 0], 0.1 * dgamma_by_dphi[:, 1], 0.1 * dgamma_by_dphi[:, 2], arrow_length_ratio=0.1, color="r")
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

    def __init__(self, order):
        self.coefficients = [np.zeros((2*order+1,)), np.zeros((2*order+1,)), np.zeros((2*order+1,))]
        self.order = order

    def num_coeff(self):
        return 3*(2*self.order+1)

    def get_dofs(self):
        return np.concatenate(self.coefficients)

    def set_dofs(self, dofs):
        counter = 0
        for i in range(3):
            self.coefficients[i][0] = dofs[counter]
            counter += 1
            for j in range(1, self.order+1):
                self.coefficients[i][2*j-1] = dofs[counter]
                counter += 1
                self.coefficients[i][2*j] = dofs[counter]
                counter += 1

    def gamma(self, points):
        res = np.zeros((len(points), 3))
        coeffs = self.coefficients
        for i in range(3):
            res[:, i] += coeffs[i][0]
            for j in range(1, self.order+1):
                res[:, i] += coeffs[i][2*j-1] * np.sin(2*pi*j*points)
                res[:, i] += coeffs[i][2*j]   * np.cos(2*pi*j*points)
        return res

    def dgamma_by_dcoeff(self, points):
        res = np.zeros((len(points), self.num_coeff(), 3))
        for i in range(3):
            res[:, i*(2*self.order+1), i] = 1
            for j in range(1, self.order+1):
                res[:, i*(2*self.order+1) + 2*j-1, i] = np.sin(2*pi*j*points)
                res[:, i*(2*self.order+1) + 2*j  , i] = np.cos(2*pi*j*points)
        return res

    def dgamma_by_dphi(self, points):
        res = np.zeros((len(points), 1, 3))
        coeffs = self.coefficients
        for i in range(3):
            for j in range(1, self.order+1):
                res[:, 0, i] += +coeffs[i][2*j-1] * 2*pi*j*np.cos(2*pi*j*points)
                res[:, 0, i] += -coeffs[i][2*j] * 2*pi*j*np.sin(2*pi*j*points)
        return res

    def d2gamma_by_dphidcoeff(self, points):
        res = np.zeros((len(points), 1, self.num_coeff(), 3))
        for i in range(3):
            for j in range(1, self.order+1):
                res[:, 0, i*(2*self.order+1) + 2*j-1, i] = +2*pi*j*np.cos(2*pi*j*points)
                res[:, 0, i*(2*self.order+1) + 2*j  , i] = -2*pi*j*np.sin(2*pi*j*points)
        return res

    def d2gamma_by_dphidphi(self, points):
        res = np.zeros((len(points), 1, 1, 3))
        coeffs = self.coefficients
        for i in range(3):
            for j in range(1, self.order+1):
                res[:, 0, 0, i] += -coeffs[i][2*j-1] * (2*pi*j)**2*np.sin(2*pi*j*points)
                res[:, 0, 0, i] += -coeffs[i][2*j]   * (2*pi*j)**2*np.cos(2*pi*j*points)
        return res

    def d3gamma_by_dphidphidcoeff(self, points):
        res = np.zeros((len(points), 1, 1, self.num_coeff(), 3))
        for i in range(3):
            for j in range(1, self.order+1):
                res[:, 0, 0, i*(2*self.order+1) + 2*j-1, i] = -(2*pi*j)**2*np.sin(2*pi*j*points)
                res[:, 0, 0, i*(2*self.order+1) + 2*j  , i] = -(2*pi*j)**2*np.cos(2*pi*j*points)
        return res



class StelleratorSymmetricCylindricalFourierCurve(Curve):

    def __init__(self, order, nfp):
        self.coefficients = [np.zeros((order+1,)), np.zeros((order,))]
        self.nfp = nfp
        self.order = order

    def num_coeff(self):
        return 2*self.order+1

    def get_dofs(self):
        return np.concatenate(self.coefficients)

    def set_dofs(self, dofs):
        counter = 0
        for i in range(self.order+1):
            self.coefficients[0][i] = dofs[i]
        for i in range(self.order):
            self.coefficients[1][i] = dofs[self.order + 1 + i]

    def gamma(self, points):
        res = np.zeros((len(points), 3))
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, 0] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            res[:, 1] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            res[:, 2] += self.coefficients[1][i-1] * np.sin(nfp * 2 * pi * i * points)
        return res

    def dgamma_by_dcoeff(self, points):
        res = np.zeros((len(points), self.num_coeff(), 3))
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, i, 0] = np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            res[:, i, 1] = np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            res[:, self.order + i, 2] = np.sin(nfp * 2 * pi * i * points)
        return res

    def dgamma_by_dphi(self, points):
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

    def d2gamma_by_dphidcoeff(self, points):
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

    def d2gamma_by_dphidphi(self, points):
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

    def d3gamma_by_dphidphidcoeff(self, points):
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
