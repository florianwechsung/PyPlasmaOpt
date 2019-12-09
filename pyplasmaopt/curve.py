from sympy import *
import numpy as np
from math import pi, sin, cos

class Curve():
    r"""
    A periodic curve \Gamma : [0, 1) \to R^3, \phi\mapsto\Gamma(\phi).
    """

    def __init__(self, points):
        if isinstance(points, int):
            self.points = np.linspace(0, 1, points, endpoint=False)
        else:
            self.points = points
        self.dependencies = []
        self.gamma = None
        self.dgamma_by_dphi = None
        self.d2gamma_by_dphidphi = None
        self.d3gamma_by_dphidphidphi = None
        self.kappa = None
        self.dgamma_by_dcoeff = None
        self.d2gamma_by_dphidcoeff = None
        self.d3gamma_by_dphidphidcoeff = None
        self.dkappa_by_dcoeff = None
        self.incremental_arclength = None
        self.dincremental_arclength_by_dcoeff = None
        self.dincremental_arclength_by_dphi = None
        self.frenet_frame = None
        self.torsion = None

    def update(self):
        num_points = len(self.points)
        num_coeffs = self.num_coeff()
        if self.gamma is None:
            self.gamma = np.zeros((num_points, 3))
        if self.dgamma_by_dphi is None:
            self.dgamma_by_dphi = np.zeros((num_points, 1, 3))
        if self.d2gamma_by_dphidphi is None:
            self.d2gamma_by_dphidphi = np.zeros((num_points, 1, 1, 3))
        if self.d3gamma_by_dphidphidphi is None:
            self.d3gamma_by_dphidphidphi = np.zeros((num_points, 1, 1, 1, 3))
        if self.kappa is None:
            self.kappa = np.zeros((num_points, 1))
        if self.dgamma_by_dcoeff is None:
            self.dgamma_by_dcoeff = np.zeros((num_points, num_coeffs, 3))
        if self.d2gamma_by_dphidcoeff is None:
            self.d2gamma_by_dphidcoeff = np.zeros((num_points, 1, num_coeffs, 3))
        if self.d3gamma_by_dphidphidcoeff is None:
            self.d3gamma_by_dphidphidcoeff = np.zeros((num_points, 1, 1, num_coeffs, 3))
        if self.dkappa_by_dcoeff is None:
            self.dkappa_by_dcoeff = np.zeros((num_points, num_coeffs, 1))
        if self.incremental_arclength is None:
            self.incremental_arclength = np.zeros((num_points, 1))
        if self.dincremental_arclength_by_dcoeff is None:
            self.dincremental_arclength_by_dcoeff = np.zeros((num_points, num_coeffs, 1))
        if self.dincremental_arclength_by_dphi is None:
            self.dincremental_arclength_by_dphi = np.zeros((num_points, 1, 1))
        if self.frenet_frame is None:
            self.frenet_frame = (np.zeros((num_points, 3)), np.zeros((num_points, 3)), np.zeros((num_points, 3)))
        if self.torsion is None:
            self.torsion = np.zeros((num_points, 1))

        self.gamma_impl()
        self.dgamma_by_dphi_impl()
        self.d2gamma_by_dphidphi_impl()
        self.d3gamma_by_dphidphidphi_impl()
        self.dgamma_by_dcoeff_impl()
        self.d2gamma_by_dphidcoeff_impl()
        self.d3gamma_by_dphidphidcoeff_impl()
        self.incremental_arclength_impl()
        self.dincremental_arclength_by_dphi_impl()
        self.dincremental_arclength_by_dcoeff_impl()
        self.kappa_impl()
        self.dkappa_by_dcoeff_impl()
        self.frenet_frame_impl()
        self.torsion_impl()
        for obj in self.dependencies:
            obj.update()

    def num_coeff(self):
        raise NotImplementedError

    def get_dofs(self):
        raise NotImplementedError

    def set_dofs(self, dofs):
        raise NotImplementedError

    def gamma_impl(self):
        """ Evaluate the curve at `points`. """
        raise NotImplementedError

    def dgamma_by_dphi_impl(self):
        """ Return the derivative of the curve. """
        raise NotImplementedError

    def d2gamma_by_dphidphi_impl(self):
        """ Return the second derivative of the curve. """
        raise NotImplementedError

    def d3gamma_by_dphidphidphi_impl(self):
        """ Return the third derivative of the curve. """
        pass

    def kappa_impl(self):
        """ Curvature at `points`. """
        points = self.points
        dgamma = self.dgamma_by_dphi[:, 0, :]
        d2gamma = self.d2gamma_by_dphidphi[:, 0, 0, :]
        self.kappa[:, :] = (np.linalg.norm(np.cross(dgamma, d2gamma), axis=1)/np.linalg.norm(dgamma, axis=1)**3).reshape(len(points),1)

    def dgamma_by_dcoeff_impl(self):
        raise NotImplementedError

    def d2gamma_by_dphidcoeff_impl(self):
        raise NotImplementedError

    def d3gamma_by_dphidphidcoeff_impl(self):
        raise NotImplementedError

    def dkappa_by_dcoeff_impl(self):
        dgamma_by_dphi = self.dgamma_by_dphi[:, 0, :]
        dgamma_by_dphidphi = self.d2gamma_by_dphidphi[:, 0, 0, :]
        dgamma_by_dphidcoeff = self.d2gamma_by_dphidcoeff[:, 0, :, :]
        dgamma_by_dphidphidcoeff = self.d3gamma_by_dphidphidcoeff[:, 0, 0, :, :]
        points = self.points
        num_coeff = dgamma_by_dphidcoeff.shape[1]
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        numerator = np.cross(dgamma_by_dphi, dgamma_by_dphidphi)
        denominator = self.incremental_arclength[:, 0]
        self.dkappa_by_dcoeff[:, :, 0] = (1 / (denominator**3*norm(numerator)))[:, None] * np.sum(numerator[:, None, :] * (
            np.cross(dgamma_by_dphidcoeff[:, :, :], dgamma_by_dphidphi[:, None, :], axis=2) +
            np.cross(dgamma_by_dphi[:, None, :], dgamma_by_dphidphidcoeff[:, :, :], axis=2)) , axis=2) \
            - (norm(numerator) * 3 / denominator**5)[:, None] * np.sum(dgamma_by_dphi[:, None, :] * dgamma_by_dphidcoeff[:, :, :], axis=2)

    def incremental_arclength_impl(self):
        self.incremental_arclength[:, :] = np.linalg.norm(self.dgamma_by_dphi[:, 0, :], axis=1).reshape((len(self.points), 1))

    def dincremental_arclength_by_dcoeff_impl(self):
        dgamma_by_dphi = self.dgamma_by_dphi[:, 0, :]
        dgamma_by_dphidcoeff = self.d2gamma_by_dphidcoeff[:, 0, :, :]
        num_coeff = dgamma_by_dphidcoeff.shape[1]
        res = self.dincremental_arclength_by_dcoeff
        res[:, :, 0] = (1/self.incremental_arclength) * np.sum(dgamma_by_dphi[:, None, :] * dgamma_by_dphidcoeff[:, :, :], axis=2)

    def dincremental_arclength_by_dphi_impl(self):
        dgamma_by_dphi = self.dgamma_by_dphi[:, 0, :]
        dgamma_by_dphidphi = self.d2gamma_by_dphidphi[:, 0, 0, :]
        res = np.zeros((len(self.points), 1, 1))
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        self.dincremental_arclength_by_dphi[:, 0, 0] = inner(dgamma_by_dphi, dgamma_by_dphidphi)/self.incremental_arclength[:,0]

    def torsion_impl(self):
        res = self.torsion
        d1gamma = self.dgamma_by_dphi[:, 0, :]
        d2gamma = self.d2gamma_by_dphidphi[:, 0, 0, :]
        d3gamma = self.d3gamma_by_dphidphidphi[:, 0, 0, 0, :]
        res[:, 0] = np.sum(np.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1) / np.sum(np.cross(d1gamma, d2gamma, axis=1)**2, axis=1)
        
    def frenet_frame_impl(self):
        """
        Returns the (t, n, b) Frenet frame.
        """
        dgamma_by_dphi = self.dgamma_by_dphi[:, 0, :]
        d2gamma_by_dphidphi = self.d2gamma_by_dphidphi[:, 0, 0, :]
        l = self.incremental_arclength
        dl_by_dphi = self.dincremental_arclength_by_dphi[:, 0, 0]
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        t, n, b = self.frenet_frame
        t[:,:] = (1./self.incremental_arclength) * dgamma_by_dphi

        tdash = (1./self.incremental_arclength)**2 * (
            self.incremental_arclength * d2gamma_by_dphidphi
            - (inner(dgamma_by_dphi, d2gamma_by_dphidphi)/self.incremental_arclength[:, 0])[:, None] *  dgamma_by_dphi
        )
        kappa = self.kappa
        n[:,:] = (1./norm(tdash))[:, None] * tdash
        b[:,:] = np.cross(t, n, axis=1)

    def plot(self, ax=None, show=True, plot_derivative=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        gamma = self.gamma
        dgamma_by_dphi = self.dgamma_by_dphi[:,0,:]
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
        counter = 0
        for i in range(3):
            self.coefficients[i][0] = dofs[counter]
            counter += 1
            for j in range(1, self.order+1):
                self.coefficients[i][2*j-1] = dofs[counter]
                counter += 1
                self.coefficients[i][2*j] = dofs[counter]
                counter += 1
        super().update()

    def gamma_impl(self):
        coeffs = self.coefficients
        points = self.points
        self.gamma *= 0
        for i in range(3):
            self.gamma[:, i] += coeffs[i][0]
            for j in range(1, self.order+1):
                self.gamma[:, i] += coeffs[i][2*j-1] * np.sin(2*pi*j*points)
                self.gamma[:, i] += coeffs[i][2*j]   * np.cos(2*pi*j*points)

    def dgamma_by_dcoeff_impl(self):
        points = self.points
        for i in range(3):
            self.dgamma_by_dcoeff[:, i*(2*self.order+1), i] = 1
            for j in range(1, self.order+1):
                self.dgamma_by_dcoeff[:, i*(2*self.order+1) + 2*j-1, i] = np.sin(2*pi*j*points)
                self.dgamma_by_dcoeff[:, i*(2*self.order+1) + 2*j  , i] = np.cos(2*pi*j*points)

    def dgamma_by_dphi_impl(self):
        coeffs = self.coefficients
        points = self.points
        self.dgamma_by_dphi *= 0
        for i in range(3):
            for j in range(1, self.order+1):
                self.dgamma_by_dphi[:, 0, i] += +coeffs[i][2*j-1] * 2*pi*j*np.cos(2*pi*j*points)
                self.dgamma_by_dphi[:, 0, i] += -coeffs[i][2*j] * 2*pi*j*np.sin(2*pi*j*points)

    def d2gamma_by_dphidcoeff_impl(self):
        points = self.points
        for i in range(3):
            for j in range(1, self.order+1):
                self.d2gamma_by_dphidcoeff[:, 0, i*(2*self.order+1) + 2*j-1, i] = +2*pi*j*np.cos(2*pi*j*points)
                self.d2gamma_by_dphidcoeff[:, 0, i*(2*self.order+1) + 2*j  , i] = -2*pi*j*np.sin(2*pi*j*points)

    def d2gamma_by_dphidphi_impl(self):
        coeffs = self.coefficients
        points = self.points
        self.d2gamma_by_dphidphi *= 0
        for i in range(3):
            for j in range(1, self.order+1):
                self.d2gamma_by_dphidphi[:, 0, 0, i] += -coeffs[i][2*j-1] * (2*pi*j)**2*np.sin(2*pi*j*points)
                self.d2gamma_by_dphidphi[:, 0, 0, i] += -coeffs[i][2*j]   * (2*pi*j)**2*np.cos(2*pi*j*points)

    def d3gamma_by_dphidphidcoeff_impl(self):
        points = self.points
        for i in range(3):
            for j in range(1, self.order+1):
                self.d3gamma_by_dphidphidcoeff[:, 0, 0, i*(2*self.order+1) + 2*j-1, i] = -(2*pi*j)**2*np.sin(2*pi*j*points)
                self.d3gamma_by_dphidphidcoeff[:, 0, 0, i*(2*self.order+1) + 2*j  , i] = -(2*pi*j)**2*np.cos(2*pi*j*points)

    def d3gamma_by_dphidphidphi_impl(self):
        coeffs = self.coefficients
        points = self.points
        self.d3gamma_by_dphidphidphi *= 0
        for i in range(3):
            for j in range(1, self.order+1):
                self.d3gamma_by_dphidphidphi[:, 0, 0, 0, i] += -coeffs[i][2*j-1] * (2*pi*j)**3*np.cos(2*pi*j*points)
                self.d3gamma_by_dphidphidphi[:, 0, 0, 0, i] += +coeffs[i][2*j]   * (2*pi*j)**3*np.sin(2*pi*j*points)


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
        counter = 0
        for i in range(self.order+1):
            self.coefficients[0][i] = dofs[i]
        for i in range(self.order):
            self.coefficients[1][i] = dofs[self.order + 1 + i]
        self.update()

    def gamma_impl(self):
        points = self.points
        res = self.gamma
        res *= 0
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, 0] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            res[:, 1] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            res[:, 2] += self.coefficients[1][i-1] * np.sin(nfp * 2 * pi * i * points)

    def dgamma_by_dcoeff_impl(self):
        res = self.dgamma_by_dcoeff
        points = self.points
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, i, 0] = np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            res[:, i, 1] = np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            res[:, self.order + i, 2] = np.sin(nfp * 2 * pi * i * points)

    def dgamma_by_dphi_impl(self):
        res = self.dgamma_by_dphi
        points = self.points
        res *= 0
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

    def d2gamma_by_dphidcoeff_impl(self):
        res = self.d2gamma_by_dphidcoeff
        nfp = self.nfp
        points = self.points
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

    def d2gamma_by_dphidphi_impl(self):
        res = self.d2gamma_by_dphidphi
        points = self.points
        res *= 0
        coeffs = self.coefficients
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, 0, 0, 0] += self.coefficients[0][i] * (
                +2*(nfp * 2 * pi * i)*(2 * pi) *       np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -((nfp * 2 * pi * i)**2 + (2*pi)**2) * np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
            res[:, 0, 0, 1] += self.coefficients[0][i] * (
                -2*(nfp * 2 * pi * i) * (2 * pi) *     np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -((nfp * 2 * pi * i)**2 + (2*pi)**2) * np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
        for i in range(1, self.order+1):
            res[:, 0, 0, 2] -= self.coefficients[1][i-1] * (nfp * 2 * pi * i)**2 * np.sin(nfp * 2 * pi * i * points)

    def d3gamma_by_dphidphidphi_impl(self):
        res = self.d3gamma_by_dphidphidphi
        points = self.points
        res *= 0
        coeffs = self.coefficients
        nfp = self.nfp
        for i in range(self.order+1):
            res[:, 0, 0, 0, 0] += self.coefficients[0][i] * (
                +2*(nfp * 2 * pi * i)**2*(2 * pi) *       np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +2*(nfp * 2 * pi * i)*(2 * pi)**2 *       np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +((nfp * 2 * pi * i)**2 + (2*pi)**2)*(nfp * 2 * pi * i)* np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +((nfp * 2 * pi * i)**2 + (2*pi)**2)*(2 * pi)*           np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
            res[:, 0, 0, 0, 1] += self.coefficients[0][i] * (
                -2*(nfp * 2 * pi * i)**2 * (2 * pi) *     np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +2*(nfp * 2 * pi * i) * (2 * pi)**2 *     np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +((nfp * 2 * pi * i)**2 + (2*pi)**2) * (nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -((nfp * 2 * pi * i)**2 + (2*pi)**2) * (2 * pi) * np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
        for i in range(1, self.order+1):
            res[:, 0, 0, 0, 2] -= self.coefficients[1][i-1] * (nfp * 2 * pi * i)**3 * np.cos(nfp * 2 * pi * i * points)

    def d3gamma_by_dphidphidcoeff_impl(self):
        res = self.d3gamma_by_dphidphidcoeff
        points = self.points
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
        self.update()

    def num_coeff(self):
        return self.curve.num_coeff()

    def gamma_impl(self):
        gamma = self.curve.gamma
        self.gamma[:, :] = gamma @ self.rotmat

    def dgamma_by_dphi_impl(self):
        dgamma_by_dphi = self.curve.dgamma_by_dphi
        self.dgamma_by_dphi[:, :, :] = dgamma_by_dphi @ self.rotmat

    def d2gamma_by_dphidphi_impl(self):
        d2gamma_by_dphidphi = self.curve.d2gamma_by_dphidphi
        self.d2gamma_by_dphidphi[:, :, :, :] = d2gamma_by_dphidphi @ self.rotmat

    def dgamma_by_dcoeff_impl(self):
        dgamma_by_dcoeff = self.curve.dgamma_by_dcoeff
        self.dgamma_by_dcoeff[:, :, :] = dgamma_by_dcoeff @ self.rotmat

    def d2gamma_by_dphidcoeff_impl(self):
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff
        self.d2gamma_by_dphidcoeff[:, :, :, :] = d2gamma_by_dphidcoeff @ self.rotmat

    def d3gamma_by_dphidphidcoeff_impl(self):
        d3gamma_by_dphidphidcoeff = self.curve.d3gamma_by_dphidphidcoeff
        self.d3gamma_by_dphidphidcoeff[:, :, :] = d3gamma_by_dphidphidcoeff @ self.rotmat

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
