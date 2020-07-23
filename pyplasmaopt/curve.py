from sympy import *
import numpy as np
from math import pi, sin, cos
from property_manager3 import cached_property, PropertyManager


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
        self.curve_properties = set([
            "gamma", "dgamma_by_dphi", "d2gamma_by_dphidphi", "d3gamma_by_dphidphidphi",
            "dgamma_by_dcoeff", "d2gamma_by_dphidcoeff", "d3gamma_by_dphidphidcoeff", "d4gamma_by_dphidphidphidcoeff",
            "kappa", "dkappa_by_dphi", "dkappa_by_dcoeff", "d2kappa_by_dphidcoeff",
            "incremental_arclength", "dincremental_arclength_by_dcoeff", "dincremental_arclength_by_dphi",
            "torsion", "dtorsion_by_dcoeff",
            "frenet_frame", "dfrenet_frame_by_dcoeff"
        ])
    def update(self):

        d = self.__dict__
        keys_to_remove = set(self.curve_properties).intersection(set(d.keys()))
        for key in keys_to_remove:
            del d[key]

        for obj in self.dependencies:
            obj.update()

    def num_coeff(self):
        raise NotImplementedError

    def get_dofs(self):
        raise NotImplementedError

    def set_dofs(self, dofs):
        raise NotImplementedError

    @cached_property
    def gamma(self):
        """ Evaluate the curve at `points`. """
        raise NotImplementedError

    @cached_property
    def dgamma_by_dphi(self):
        """ Return the derivative of the curve. """
        raise NotImplementedError

    @cached_property
    def d2gamma_by_dphidphi(self):
        """ Return the second derivative of the curve. """
        raise NotImplementedError

    @cached_property
    def d3gamma_by_dphidphidphi(self):
        """ Return the third derivative of the curve. """
        pass

    @cached_property
    def kappa(self):
        """ Curvature at `points`. """
        kappa = np.zeros((len(self.points), 1))
        points = self.points
        dgamma = self.dgamma_by_dphi[:, 0, :]
        d2gamma = self.d2gamma_by_dphidphi[:, 0, 0, :]
        kappa[:, :] = (np.linalg.norm(np.cross(dgamma, d2gamma), axis=1)/np.linalg.norm(dgamma, axis=1)**3).reshape(len(points),1)
        return kappa

    @cached_property
    def dkappa_by_dphi(self):
        dkappa_by_dphi = np.zeros((len(self.points), 1, 1))
        points = self.points
        dgamma = self.dgamma_by_dphi[:, 0, :]
        d2gamma = self.d2gamma_by_dphidphi[:, 0, 0, :]
        d3gamma = self.d3gamma_by_dphidphidphi[:, 0, 0, 0, :]
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        cross = lambda a, b: np.cross(a, b, axis=1)
        dkappa_by_dphi[:, 0, 0] = inner(cross(dgamma, d2gamma), cross(dgamma, d3gamma))/(norm(cross(dgamma, d2gamma)) * norm(dgamma)**3) \
            - 3 * inner(dgamma, d2gamma) * norm(cross(dgamma, d2gamma))/norm(dgamma)**5
        return dkappa_by_dphi

    @cached_property
    def d2kappa_by_dphidcoeff(self):
        d2kappa_by_dphidcoeff = np.zeros((len(self.points), 1, self.num_coeff(), 1))
        points = self.points
        dgamma = self.dgamma_by_dphi[:, 0, :]
        d2gamma = self.d2gamma_by_dphidphi[:, 0, 0, :]
        d3gamma = self.d3gamma_by_dphidphidphi[:, 0, 0, 0, :]

        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        cross = lambda a, b: np.cross(a, b, axis=1)
        d1_dot_d2 = inner(dgamma, d2gamma)
        d1_x_d2   = cross(dgamma, d2gamma)
        d1_x_d3   = cross(dgamma, d3gamma)
        normdgamma = norm(dgamma)
        for i in range(self.num_coeff()):
            dgamma_dcoeff  = self.d2gamma_by_dphidcoeff[:, 0, i, :]
            d2gamma_dcoeff = self.d3gamma_by_dphidphidcoeff[:, 0, 0, i, :]
            d3gamma_dcoeff = self.d4gamma_by_dphidphidphidcoeff[:, 0, 0, 0, i, :]

            d1coeff_x_d2   = cross(dgamma_dcoeff, d2gamma)
            d1coeff_dot_d2 = inner(dgamma_dcoeff, d2gamma)
            d1coeff_x_d3   = cross(dgamma_dcoeff, d3gamma)
            d1_x_d2coeff   = cross(dgamma, d2gamma_dcoeff)
            d1_dot_d2coeff = inner(dgamma, d2gamma_dcoeff)
            d1_dot_d1coeff = inner(dgamma, dgamma_dcoeff)
            d1_x_d3coeff   = cross(dgamma, d3gamma_dcoeff)

            d2kappa_by_dphidcoeff[:, 0, i, 0] = (
                +inner(d1coeff_x_d2 + d1_x_d2coeff, d1_x_d3)
                +inner(d1_x_d2, d1coeff_x_d3 + d1_x_d3coeff)
            )/(norm(d1_x_d2) * normdgamma**3) \
                -inner(d1_x_d2, d1_x_d3) * (
                    (
                        inner(d1coeff_x_d2 + d1_x_d2coeff, d1_x_d2)/(norm(d1_x_d2)**3 * normdgamma**3)
                        + 3 * inner(dgamma, dgamma_dcoeff)/(norm(d1_x_d2) * normdgamma**5)
                    )
                ) \
                - 3 * (
                    + (d1coeff_dot_d2 + d1_dot_d2coeff) * norm(d1_x_d2)/normdgamma**5
                    + d1_dot_d2 * inner(d1coeff_x_d2 + d1_x_d2coeff, d1_x_d2)/(norm(d1_x_d2) * normdgamma**5)
                    - 5 * d1_dot_d2 * norm(d1_x_d2) * d1_dot_d1coeff/normdgamma**7
                )
        return d2kappa_by_dphidcoeff

    @cached_property
    def d2gamma_by_dphidcoeff(self):
        raise NotImplementedError

    @cached_property
    def d3gamma_by_dphidphidcoeff(self):
        raise NotImplementedError

    @cached_property
    def d4gamma_by_dphidphidphidcoeff(self):
        raise NotImplementedError

    @cached_property
    def dkappa_by_dcoeff(self):
        dkappa_by_dcoeff = np.zeros((len(self.points), self.num_coeff(), 1))
        dgamma_by_dphi = self.dgamma_by_dphi[:, 0, :]
        dgamma_by_dphidphi = self.d2gamma_by_dphidphi[:, 0, 0, :]
        dgamma_by_dphidcoeff = self.d2gamma_by_dphidcoeff[:, 0, :, :]
        dgamma_by_dphidphidcoeff = self.d3gamma_by_dphidphidcoeff[:, 0, 0, :, :]

        num_coeff = dgamma_by_dphidcoeff.shape[1]
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        numerator = np.cross(dgamma_by_dphi, dgamma_by_dphidphi)
        denominator = self.incremental_arclength[:, 0]
        dkappa_by_dcoeff[:, :, 0] = (1 / (denominator**3*norm(numerator)))[:, None] * np.sum(numerator[:, None, :] * (
            np.cross(dgamma_by_dphidcoeff[:, :, :], dgamma_by_dphidphi[:, None, :], axis=2) +
            np.cross(dgamma_by_dphi[:, None, :], dgamma_by_dphidphidcoeff[:, :, :], axis=2)) , axis=2) \
            - (norm(numerator) * 3 / denominator**5)[:, None] * np.sum(dgamma_by_dphi[:, None, :] * dgamma_by_dphidcoeff[:, :, :], axis=2)
        return dkappa_by_dcoeff

    @cached_property
    def incremental_arclength(self):
        incremental_arclength = np.zeros((len(self.points), 1))
        incremental_arclength[:, :] = np.linalg.norm(self.dgamma_by_dphi[:, 0, :], axis=1).reshape((len(self.points), 1))
        return incremental_arclength

    @cached_property
    def dincremental_arclength_by_dcoeff(self):
        dgamma_by_dphi = self.dgamma_by_dphi[:, 0, :]
        dgamma_by_dphidcoeff = self.d2gamma_by_dphidcoeff[:, 0, :, :]
        num_coeff = dgamma_by_dphidcoeff.shape[1]
        res = np.zeros((len(self.points), self.num_coeff(), 1))
        res[:, :, 0] = (1/self.incremental_arclength) * np.sum(dgamma_by_dphi[:, None, :] * dgamma_by_dphidcoeff[:, :, :], axis=2)
        return res

    @cached_property
    def dincremental_arclength_by_dphi(self):
        dincremental_arclength_by_dphi = np.zeros((len(self.points), 1, 1))
        dgamma_by_dphi = self.dgamma_by_dphi[:, 0, :]
        dgamma_by_dphidphi = self.d2gamma_by_dphidphi[:, 0, 0, :]
        res = np.zeros((len(self.points), 1, 1))
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        dincremental_arclength_by_dphi[:, 0, 0] = inner(dgamma_by_dphi, dgamma_by_dphidphi)/self.incremental_arclength[:,0]
        return dincremental_arclength_by_dphi 

    @cached_property
    def torsion(self):
        torsion = np.zeros((len(self.points), 1))
        d1gamma = self.dgamma_by_dphi[:, 0, :]
        d2gamma = self.d2gamma_by_dphidphi[:, 0, 0, :]
        d3gamma = self.d3gamma_by_dphidphidphi[:, 0, 0, 0, :]
        torsion[:, 0] = np.sum(np.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1) / np.sum(np.cross(d1gamma, d2gamma, axis=1)**2, axis=1)
        return torsion

    @cached_property
    def dtorsion_by_dcoeff(self):
        dtorsion_by_dcoeff = np.zeros((len(self.points), self.num_coeff(), 1))
        d1gamma = self.dgamma_by_dphi[:, 0, :]
        d2gamma = self.d2gamma_by_dphidphi[:, 0, 0, :]
        d3gamma = self.d3gamma_by_dphidphidphi[:, 0, 0, 0, :]
        d1gammadcoeff = self.d2gamma_by_dphidcoeff[:, 0, :]
        d2gammadcoeff = self.d3gamma_by_dphidphidcoeff[:, 0, 0, :, :]
        d3gammadcoeff = self.d4gamma_by_dphidphidphidcoeff[:, 0, 0, 0, :, :]
        dtorsion_by_dcoeff[:, :, 0] = (
              np.sum(np.cross(d1gamma, d2gamma, axis=1)[:, None, :] * d3gammadcoeff, axis=2)
            + np.sum((np.cross(d1gammadcoeff, d2gamma[:, None, :], axis=2) +  np.cross(d1gamma[:, None, :], d2gammadcoeff, axis=2)) * d3gamma[:, None, :], axis=2)
        )/np.sum(np.cross(d1gamma, d2gamma, axis=1)**2, axis=1)[:, None]
        dtorsion_by_dcoeff[:, :, 0] -= np.sum(np.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1)[:, None] * np.sum(2 * np.cross(d1gamma, d2gamma, axis=1)[:, None, :] * (np.cross(d1gammadcoeff, d2gamma[:, None, :], axis=2) + np.cross(d1gamma[:, None, :], d2gammadcoeff, axis=2)), axis=2)/np.sum(np.cross(d1gamma, d2gamma, axis=1)**2, axis=1)[:, None]**2
        return dtorsion_by_dcoeff
        
    @cached_property
    def frenet_frame(self):
        """
        Returns the (t, n, b) Frenet frame.
        """
        dgamma_by_dphi = self.dgamma_by_dphi[:, 0, :]
        d2gamma_by_dphidphi = self.d2gamma_by_dphidphi[:, 0, 0, :]
        l = self.incremental_arclength
        norm = lambda a: np.linalg.norm(a, axis=1)
        inner = lambda a, b: np.sum(a*b, axis=1)
        t, n, b = (np.zeros((len(self.points), 3)), np.zeros((len(self.points), 3)), np.zeros((len(self.points), 3)))
        t[:,:] = (1./self.incremental_arclength) * dgamma_by_dphi

        tdash = (1./self.incremental_arclength)**2 * (
            self.incremental_arclength * d2gamma_by_dphidphi
            - (inner(dgamma_by_dphi, d2gamma_by_dphidphi)/self.incremental_arclength[:, 0])[:, None] *  dgamma_by_dphi
        )
        kappa = self.kappa
        n[:,:] = (1./norm(tdash))[:, None] * tdash
        # n[:,:] = tdash
        b[:,:] = np.cross(t, n, axis=1)
        return t, n, b

    @cached_property
    def dfrenet_frame_by_dcoeff(self):
        dgamma_by_dphi            = self.dgamma_by_dphi[:, 0, :]
        d2gamma_by_dphidphi       = self.d2gamma_by_dphidphi[:, 0, 0, :]
        d2gamma_by_dphidcoeff     = self.d2gamma_by_dphidcoeff[:, 0, :, :]
        d3gamma_by_dphidphidcoeff = self.d3gamma_by_dphidphidcoeff[:, 0, 0, :, :]

        l = self.incremental_arclength
        dl_by_dcoeff = self.dincremental_arclength_by_dcoeff

        norm   = lambda a: np.linalg.norm(a, axis=1)
        inner  = lambda a, b: np.sum(a*b, axis=1)
        inner2 = lambda a, b: np.sum(a*b, axis=2)

        dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff = (np.zeros((len(self.points), self.num_coeff(), 3)), np.zeros((len(self.points), self.num_coeff(), 3)), np.zeros((len(self.points), self.num_coeff(), 3)))
        t, n, b = self.frenet_frame

        dt_by_dcoeff[:, :, :] = -(dl_by_dcoeff/l[:, None, :]**2) * dgamma_by_dphi[:, None, :] \
            + d2gamma_by_dphidcoeff / l[:, None, :]

        tdash = (1./l)**2 * (
            l * d2gamma_by_dphidphi
            - (inner(dgamma_by_dphi, d2gamma_by_dphidphi)/l[:, 0])[:, None] *  dgamma_by_dphi
        )

        dtdash_by_dcoeff = (-2 * dl_by_dcoeff / l[:, None, :]**3) * (l * d2gamma_by_dphidphi - (inner(dgamma_by_dphi, d2gamma_by_dphidphi)/l[:, 0])[:, None] *  dgamma_by_dphi)[:, None, :] \
            + (1./l[:, None, :])**2 * (
                dl_by_dcoeff * d2gamma_by_dphidphi[:, None, :] + l[:, None, :] * d3gamma_by_dphidphidcoeff
                - (inner2(d2gamma_by_dphidcoeff, d2gamma_by_dphidphi[:, None, :])[:, :, None]/l[:, None, :]) *  dgamma_by_dphi[:, None, :]
                - (inner2(dgamma_by_dphi[:, None, :], d3gamma_by_dphidphidcoeff)[:, :, None]/l[:, None, :]) *  dgamma_by_dphi[:, None, :]
                + (inner(dgamma_by_dphi, d2gamma_by_dphidphi)[:, None, None] * dl_by_dcoeff/l[:, None, :]**2) *  dgamma_by_dphi[:, None, :]
                - (inner(dgamma_by_dphi, d2gamma_by_dphidphi)[:, None]/l)[:, None, :] *  d2gamma_by_dphidcoeff
            )
        dn_by_dcoeff[:, :, :] = (1./norm(tdash))[:, None, None] * dtdash_by_dcoeff \
            - (inner2(tdash[:, None, :], dtdash_by_dcoeff)[:, :, None]/inner(tdash, tdash)[:, None, None]**1.5) * tdash[:, None, :]

        db_by_dcoeff[:, :, :] = np.cross(dt_by_dcoeff, n[:, None, :], axis=2) + np.cross(t[:, None, :], dn_by_dcoeff, axis=2)
        return dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff

    def plot(self, ax=None, show=True, plot_derivative=False, closed_loop=True, color=None, linestyle=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        gamma = self.gamma
        dgamma_by_dphi = self.dgamma_by_dphi[:,0,:]
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        def rep(data):
            if closed_loop:
                return np.concatenate((data, [data[0]]))
            else:
                return data
        ax.plot(rep(gamma[:, 0]), rep(gamma[:, 1]), rep(gamma[:, 2]), color=color, linestyle=linestyle)
        if plot_derivative:
            ax.quiver(rep(gamma[:, 0]), rep(gamma[:, 1]), rep(gamma[:, 2]), 0.1 * rep(dgamma_by_dphi[:, 0]), 0.1 * rep(dgamma_by_dphi[:, 1]), 0.1 * rep(dgamma_by_dphi[:, 2]), arrow_length_ratio=0.1, color="r")
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

    @cached_property
    def gamma(self):
        gamma = np.zeros((len(self.points), 3))
        coeffs = self.coefficients
        points = self.points
        for i in range(3):
            gamma[:, i] += coeffs[i][0]
            for j in range(1, self.order+1):
                gamma[:, i] += coeffs[i][2*j-1] * np.sin(2*pi*j*points)
                gamma[:, i] += coeffs[i][2*j]   * np.cos(2*pi*j*points)
        return gamma

    @cached_property
    def dgamma_by_dcoeff(self):
        dgamma_by_dcoeff = np.zeros((len(self.points), self.num_coeff(), 3))
        points = self.points
        for i in range(3):
            dgamma_by_dcoeff[:, i*(2*self.order+1), i] = 1
            for j in range(1, self.order+1):
                dgamma_by_dcoeff[:, i*(2*self.order+1) + 2*j-1, i] = np.sin(2*pi*j*points)
                dgamma_by_dcoeff[:, i*(2*self.order+1) + 2*j  , i] = np.cos(2*pi*j*points)
        return dgamma_by_dcoeff

    @cached_property
    def dgamma_by_dphi(self):
        dgamma_by_dphi = np.zeros((len(self.points), 1, 3))
        coeffs = self.coefficients
        points = self.points
        for i in range(3):
            for j in range(1, self.order+1):
                dgamma_by_dphi[:, 0, i] += +coeffs[i][2*j-1] * 2*pi*j*np.cos(2*pi*j*points)
                dgamma_by_dphi[:, 0, i] += -coeffs[i][2*j] * 2*pi*j*np.sin(2*pi*j*points)
        return dgamma_by_dphi

    @cached_property
    def d2gamma_by_dphidcoeff(self):
        d2gamma_by_dphidcoeff = np.zeros((len(self.points), 1, self.num_coeff(), 3))
        points = self.points
        for i in range(3):
            for j in range(1, self.order+1):
                d2gamma_by_dphidcoeff[:, 0, i*(2*self.order+1) + 2*j-1, i] = +2*pi*j*np.cos(2*pi*j*points)
                d2gamma_by_dphidcoeff[:, 0, i*(2*self.order+1) + 2*j  , i] = -2*pi*j*np.sin(2*pi*j*points)
        return d2gamma_by_dphidcoeff

    @cached_property
    def d2gamma_by_dphidphi(self):
        d2gamma_by_dphidphi = np.zeros((len(self.points), 1, 1, 3))
        coeffs = self.coefficients
        points = self.points
        for i in range(3):
            for j in range(1, self.order+1):
                d2gamma_by_dphidphi[:, 0, 0, i] += -coeffs[i][2*j-1] * (2*pi*j)**2*np.sin(2*pi*j*points)
                d2gamma_by_dphidphi[:, 0, 0, i] += -coeffs[i][2*j]   * (2*pi*j)**2*np.cos(2*pi*j*points)
        return d2gamma_by_dphidphi

    @cached_property
    def d3gamma_by_dphidphidcoeff(self):
        d3gamma_by_dphidphidcoeff = np.zeros((len(self.points), 1, 1, self.num_coeff(), 3))
        points = self.points
        for i in range(3):
            for j in range(1, self.order+1):
                d3gamma_by_dphidphidcoeff[:, 0, 0, i*(2*self.order+1) + 2*j-1, i] = -(2*pi*j)**2*np.sin(2*pi*j*points)
                d3gamma_by_dphidphidcoeff[:, 0, 0, i*(2*self.order+1) + 2*j  , i] = -(2*pi*j)**2*np.cos(2*pi*j*points)
        return d3gamma_by_dphidphidcoeff

    @cached_property
    def d3gamma_by_dphidphidphi(self):
        d3gamma_by_dphidphidphi = np.zeros((len(self.points), 1, 1, 1, 3))
        coeffs = self.coefficients
        points = self.points
        for i in range(3):
            for j in range(1, self.order+1):
                d3gamma_by_dphidphidphi[:, 0, 0, 0, i] += -coeffs[i][2*j-1] * (2*pi*j)**3*np.cos(2*pi*j*points)
                d3gamma_by_dphidphidphi[:, 0, 0, 0, i] += +coeffs[i][2*j]   * (2*pi*j)**3*np.sin(2*pi*j*points)
        return d3gamma_by_dphidphidphi

    @cached_property
    def d4gamma_by_dphidphidphidcoeff(self):
        d4gamma_by_dphidphidphidcoeff = np.zeros((len(self.points), 1, 1, 1, self.num_coeff(), 3))
        points = self.points
        for i in range(3):
            for j in range(1, self.order+1):
                d4gamma_by_dphidphidphidcoeff[:, 0, 0, 0, i*(2*self.order+1) + 2*j-1, i] = -(2*pi*j)**3*np.cos(2*pi*j*points)
                d4gamma_by_dphidphidphidcoeff[:, 0, 0, 0, i*(2*self.order+1) + 2*j  , i] = +(2*pi*j)**3*np.sin(2*pi*j*points)
        return d4gamma_by_dphidphidphidcoeff

#Class used to define the magnetic axis in example 2
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

    @cached_property
    def gamma(self):
        gamma = np.zeros((len(self.points), 3))
        points = self.points
        nfp = self.nfp
        for i in range(self.order+1):
            gamma[:, 0] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            gamma[:, 1] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            gamma[:, 2] += self.coefficients[1][i-1] * np.sin(nfp * 2 * pi * i * points)
        return gamma

    @cached_property
    def dgamma_by_dcoeff(self):
        dgamma_by_dcoeff = np.zeros((len(self.points), self.num_coeff(), 3))
        points = self.points
        nfp = self.nfp
        for i in range(self.order+1):
            dgamma_by_dcoeff[:, i, 0] = np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            dgamma_by_dcoeff[:, i, 1] = np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            dgamma_by_dcoeff[:, self.order + i, 2] = np.sin(nfp * 2 * pi * i * points)
        return dgamma_by_dcoeff

    @cached_property
    def dgamma_by_dphi(self):
        dgamma_by_dphi = np.zeros((len(self.points), 1, 3))
        points = self.points
        nfp = self.nfp
        for i in range(self.order+1):
            dgamma_by_dphi[:, 0, 0] += self.coefficients[0][i] * (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
            dgamma_by_dphi[:, 0, 1] += self.coefficients[0][i] * (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
        for i in range(1, self.order+1):
            dgamma_by_dphi[:, 0, 2] += self.coefficients[1][i-1] * (nfp * 2 * pi * i) * np.cos(nfp * 2 * pi * i * points)
        return dgamma_by_dphi

    @cached_property
    def d2gamma_by_dphidcoeff(self):
        d2gamma_by_dphidcoeff = np.zeros((len(self.points), 1, self.num_coeff(), 3))
        nfp = self.nfp
        points = self.points
        for i in range(self.order+1):
            d2gamma_by_dphidcoeff[:, 0, i, 0] = (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
            d2gamma_by_dphidcoeff[:, 0, i, 1] = (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
        for i in range(1, self.order+1):
            d2gamma_by_dphidcoeff[:, 0, self.order + i, 2] = (nfp * 2 * pi * i) * np.cos(nfp * 2 * pi * i * points)
        return d2gamma_by_dphidcoeff

    @cached_property
    def d2gamma_by_dphidphi(self):
        d2gamma_by_dphidphi = np.zeros((len(self.points), 1, 1, 3))
        points = self.points
        coeffs = self.coefficients
        nfp = self.nfp
        for i in range(self.order+1):
            d2gamma_by_dphidphi[:, 0, 0, 0] += self.coefficients[0][i] * (
                +2*(nfp * 2 * pi * i)*(2 * pi) *       np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -((nfp * 2 * pi * i)**2 + (2*pi)**2) * np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
            d2gamma_by_dphidphi[:, 0, 0, 1] += self.coefficients[0][i] * (
                -2*(nfp * 2 * pi * i) * (2 * pi) *     np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -((nfp * 2 * pi * i)**2 + (2*pi)**2) * np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
        for i in range(1, self.order+1):
            d2gamma_by_dphidphi[:, 0, 0, 2] -= self.coefficients[1][i-1] * (nfp * 2 * pi * i)**2 * np.sin(nfp * 2 * pi * i * points)
        return d2gamma_by_dphidphi

    @cached_property
    def d3gamma_by_dphidphidphi(self):
        d3gamma_by_dphidphidphi = np.zeros((len(self.points), 1, 1, 1, 3))
        points = self.points
        coeffs = self.coefficients
        nfp = self.nfp
        for i in range(self.order+1):
            d3gamma_by_dphidphidphi[:, 0, 0, 0, 0] += self.coefficients[0][i] * (
                +2*(nfp * 2 * pi * i)**2*(2 * pi) *       np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +2*(nfp * 2 * pi * i)*(2 * pi)**2 *       np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +((nfp * 2 * pi * i)**2 + (2*pi)**2)*(nfp * 2 * pi * i)* np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +((nfp * 2 * pi * i)**2 + (2*pi)**2)*(2 * pi)*           np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
            d3gamma_by_dphidphidphi[:, 0, 0, 0, 1] += self.coefficients[0][i] * (
                -2*(nfp * 2 * pi * i)**2 * (2 * pi) *     np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +2*(nfp * 2 * pi * i) * (2 * pi)**2 *     np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +((nfp * 2 * pi * i)**2 + (2*pi)**2) * (nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -((nfp * 2 * pi * i)**2 + (2*pi)**2) * (2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
        for i in range(1, self.order+1):
            d3gamma_by_dphidphidphi[:, 0, 0, 0, 2] -= self.coefficients[1][i-1] * (nfp * 2 * pi * i)**3 * np.cos(nfp * 2 * pi * i * points)
        return d3gamma_by_dphidphidphi

    @cached_property
    def d3gamma_by_dphidphidcoeff(self):
        d3gamma_by_dphidphidcoeff = np.zeros((len(self.points), 1, 1, self.num_coeff(), 3))
        points = self.points
        nfp = self.nfp
        for i in range(self.order+1):
            d3gamma_by_dphidphidcoeff[:, 0, 0, i, 0] = (
                -(nfp * 2 * pi * i)**2 *       np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +(nfp * 2 * pi * i)*(2 * pi) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +(nfp * 2 * pi * i)*(2 * pi) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -(2 * pi)**2 *                 np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
            d3gamma_by_dphidphidcoeff[:, 0, 0, i, 1] = (
                -(nfp * 2 * pi * i)**2 *         np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -(nfp * 2 * pi * i) * (2 * pi) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi) * (nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi)**2 *                   np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
        for i in range(1, self.order+1):
            d3gamma_by_dphidphidcoeff[:, 0, 0, self.order + i, 2] = -(nfp * 2 * pi * i)**2 * np.sin(nfp * 2 * pi * i * points)
        return d3gamma_by_dphidphidcoeff

    @cached_property
    def d4gamma_by_dphidphidphidcoeff(self):
        d4gamma_by_dphidphidphidcoeff = np.zeros((len(self.points), 1, 1, 1, self.num_coeff(), 3))
        points = self.points
        nfp = self.nfp
        for i in range(self.order+1):
            d4gamma_by_dphidphidphidcoeff[:, 0, 0, 0, i, 0] = (
                +2*(nfp * 2 * pi * i)**2*(2 * pi) *       np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +2*(nfp * 2 * pi * i)*(2 * pi)**2 *       np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +((nfp * 2 * pi * i)**2 + (2*pi)**2)*(nfp * 2 * pi * i)* np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +((nfp * 2 * pi * i)**2 + (2*pi)**2)*(2 * pi)*           np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
            d4gamma_by_dphidphidphidcoeff[:, 0, 0, 0, i, 1] = (
                -2*(nfp * 2 * pi * i)**2 * (2 * pi) *     np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                +2*(nfp * 2 * pi * i) * (2 * pi)**2 *     np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +((nfp * 2 * pi * i)**2 + (2*pi)**2) * (nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                -((nfp * 2 * pi * i)**2 + (2*pi)**2) * (2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
        for i in range(1, self.order+1):
            d4gamma_by_dphidphidphidcoeff[:, 0, 0, 0, self.order + i, 2] = -(nfp * 2 * pi * i)**3 * np.cos(nfp * 2 * pi * i * points)
        return d4gamma_by_dphidphidphidcoeff

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

    @cached_property
    def gamma(self):
        return self.curve.gamma @ self.rotmat

    @cached_property
    def dgamma_by_dphi(self):
        return self.curve.dgamma_by_dphi @ self.rotmat

    @cached_property
    def d2gamma_by_dphidphi(self):
        return self.curve.d2gamma_by_dphidphi @ self.rotmat

    @cached_property
    def dgamma_by_dcoeff(self):
        return self.curve.dgamma_by_dcoeff @ self.rotmat

    @cached_property
    def d2gamma_by_dphidcoeff(self):
        return self.curve.d2gamma_by_dphidcoeff @ self.rotmat

    @cached_property
    def d3gamma_by_dphidphidcoeff(self):
        return self.curve.d3gamma_by_dphidphidcoeff @ self.rotmat

    @cached_property
    def d4gamma_by_dphidphidphidcoeff(self):
        return self.curve.d4gamma_by_dphidphidphidcoeff @ self.rotmat


class GaussianSampler():

    def __init__(self, points, sigma, length_scale):
        self.points = points
        xs = self.points
        n = len(xs)
        n_derivs = 3
        self.n_derivs = n_derivs
        cov_mat = np.zeros((n*(n_derivs+1), n*(n_derivs+1)))
        def kernel(x, y):
            return sum(sigma**2*exp(-(x-y+i)**2/length_scale**2) for i in range(-2, 3))
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
        self.L = np.real(sqrtm(cov_mat))

    def sample(self, randomgen=None):
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return curve_and_derivs[0:n, :], curve_and_derivs[n:2*n, :], \
            curve_and_derivs[2*n:3*n, :], curve_and_derivs[3*n:4*n, :]


class GaussianPerturbedCurve(Curve):

    def __init__(self, curve, sampler, randomgen=None):
        super().__init__(curve.points)
        self.curve = curve
        self.sampler = sampler
        curve.dependencies.append(self)
        self.randomgen = randomgen
        self.sample = sampler.sample(self.randomgen)

    def resample(self):
        self.sample = self.sampler.sample(self.randomgen)
        self.update()

    def num_coeff(self):
        return self.curve.num_coeff()

    def get_dofs(self):
        return self.curve.get_dofs()

    def set_dofs(self, x):
        return self.curve.set_dofs(x)

    @cached_property
    def gamma(self):
        return self.curve.gamma + self.sample[0]

    @cached_property
    def dgamma_by_dphi(self):
        return self.curve.dgamma_by_dphi + self.sample[1][:, None, :]

    @cached_property
    def d2gamma_by_dphidphi(self):
        return self.curve.d2gamma_by_dphidphi + self.sample[2][:, None, None, :]

    @cached_property
    def d3gamma_by_dphidphidphi(self):
        return self.curve.d3gamma_by_dphidphidphi + self.sample[3][:, None, None, None, :]

    @cached_property
    def dgamma_by_dcoeff(self):
        return self.curve.dgamma_by_dcoeff

    @cached_property
    def d2gamma_by_dphidcoeff(self):
        return self.curve.d2gamma_by_dphidcoeff

    @cached_property
    def d3gamma_by_dphidphidcoeff(self):
        return self.curve.d3gamma_by_dphidphidcoeff

    @cached_property
    def d4gamma_by_dphidphidphidcoeff(self):
        return self.curve.d4gamma_by_dphidphidphidcoeff
