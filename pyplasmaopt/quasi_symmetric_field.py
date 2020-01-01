from .curve import Curve
import numpy as np
from math import pi
from scipy.optimize import fsolve
from property_manager3 import cached_property, PropertyManager
writable_cached_property = cached_property(writable=True)


class QuasiSymmetricField(PropertyManager):

    def __init__(self, eta_bar, magnetic_axis):
        self.s_G = 1
        self.B_0 = 1
        self.s_Psi = 1
        self.eta_bar = eta_bar
        self.magnetic_axis = magnetic_axis
        self.n = len(magnetic_axis.points)
        self.__state = np.zeros((self.n+1,))
        import scipy
        n = self.n
        points = self.magnetic_axis.points.reshape((n, 1))
        oneton = np.asarray(range(0, n)).reshape((n, 1))
        fak = (2 * pi) / (points[-1] - points[0] + (points[1]-points[0]))
        dists = fak * scipy.spatial.distance.cdist(points, points, lambda a, b: a-b)
        np.fill_diagonal(dists, 1e-10) # to shut up the warning
        if n % 2 == 0:
            D = 0.5 \
                * np.power(-1, scipy.spatial.distance.cdist(oneton, -oneton)) \
                / np.tan(0.5 * dists)
        else:
            D = 0.5 \
                * np.power(-1, scipy.spatial.distance.cdist(oneton, -oneton)) \
                / np.sin(0.5 * dists)
        
        np.fill_diagonal(D, 0)
        D *=  fak
        self.D = D

    def clear(self):
        self.clear_cached_properties()

    @writable_cached_property
    def B(self):
        self.compute_forward()
        return self.B

    @writable_cached_property
    def dB_by_dX(self):
        self.compute_forward()
        return self.dB_by_dX

    @writable_cached_property
    def sigma(self):
        self.compute_forward()
        return self.sigma

    @writable_cached_property
    def iota(self):
        self.compute_forward()
        return self.iota

    @writable_cached_property
    def dsigma_by_dphi(self):
        self.compute_forward()
        return self.dsigma_by_dphi

    def compute_forward(self):
        sigma, iota, dsigma_by_dphi = self.solve_state()
        self.sigma = sigma
        self.iota = iota
        self.dsigma_by_dphi = dsigma_by_dphi
        (t, n, b) = self.magnetic_axis.frenet_frame
        """ Compute B """
        self.B = self.B_0 * t
        """ Compute dB_by_dX """
        kappa = self.magnetic_axis.kappa[:,0]
        dkappa_by_dphi = self.magnetic_axis.dkappa_by_dphi[:,0,0]
        torsion = self.magnetic_axis.torsion[:,0]
        ldash = self.magnetic_axis.incremental_arclength[:,0]
        s_Psi = self.s_Psi
        s_G = self.s_G
        B_0 = self.B_0
        G_0 = np.mean(ldash) * self.s_G * self.B_0/(2*pi)
        eta_bar = self.eta_bar
        iota = self.iota
        sigma = self.sigma
        dsigma_dphi = self.dsigma_by_dphi
        X1c = eta_bar/kappa
        Y1s = s_G * s_Psi * kappa / eta_bar
        Y1c = s_G * s_Psi * kappa * sigma / eta_bar
        dX1c_dphi = -eta_bar * dkappa_by_dphi / kappa**2
        dY1s_dphi = s_G * s_Psi * dkappa_by_dphi / eta_bar
        dY1c_dphi = s_G * s_Psi * (dkappa_by_dphi * sigma + kappa * dsigma_dphi) / eta_bar
        dX1c_dvarphi = abs(G_0) * dX1c_dphi/(ldash * B_0)
        dY1s_dvarphi = abs(G_0) * dY1s_dphi/(ldash * B_0)
        dY1c_dvarphi = abs(G_0) * dY1c_dphi/(ldash * B_0)
        self.dB_by_dX = np.zeros((self.n, 3, 3))
        for j in range(3):
            nterm = s_Psi * G_0 * kappa * t[:, j] / B_0
            nterm += (dX1c_dvarphi * Y1s + iota * X1c * Y1c) * n[:, j]
            nterm += (dY1c_dvarphi * Y1s - dY1s_dvarphi * Y1c + s_Psi * G_0 * B_0 * torsion + iota*(Y1s**2 + Y1c**2)) * b[:, j]
            bterm = (-s_Psi * G_0 * torsion/B_0 - iota * X1c**2) * n[:, j]
            bterm += (X1c * dY1s_dvarphi - iota * X1c * Y1c) * b[:, j]
            tterm = kappa * s_G * B_0 * n[:, j]
            self.dB_by_dX[:, j, :] = s_Psi * (B_0**2/abs(G_0)) * (nterm[:, None] * n + bterm[:, None] * b) + tterm[:, None] * t

    def solve_state(self):
        n = self.n
        ldash = self.magnetic_axis.incremental_arclength[:, 0]
        kappa = self.magnetic_axis.kappa[:, 0]

        G_0  = np.mean(ldash) * self.s_G * self.B_0/(2*pi)
        fak1 = abs(G_0)/self.B_0
        fak2 = 2 * G_0 * self.eta_bar**2 / (self.s_Psi * self.B_0)
        torsion = self.magnetic_axis.torsion[:, 0]

        def build_residual(x):
            sigma = x[:-1]
            iota = x[-1]
            residual = np.zeros((n+1, ))
            residual[:n] = (fak1/ldash)*(self.D@sigma) + iota * ((self.eta_bar/kappa)**4 + 1 + sigma**2) + fak2 * torsion / kappa**2
            residual[-1] = sigma[0]
            return residual

        def build_jacobian(x):
            sigma = x[:-1]
            iota = x[-1]
            jacobian = np.zeros((n+1, n+1))
            jacobian[:n, :n] = np.diag(fak1/ldash)@self.D + np.diag(2 * sigma * iota)
            jacobian[:n, n] = ((self.eta_bar/kappa)**4 + 1 + sigma**2)
            jacobian[-1, 0] = 1
            return jacobian

        # x = np.random.rand(*self.state.shape)
        # jac = build_jacobian(x)
        # jac_est = np.zeros(jac.shape)
        # f0 = build_residual(x)
        # eps = 1e-4
        # for i in range(self.n+1):
        #     x[i] += eps
        #     fx = build_residual(x)
        #     x[i] -= 2*eps
        #     fy = build_residual(x)
        #     x[i] += eps
        #     jac_est[:, i] = (fx-fy)/(2*eps)
        # np.set_printoptions(linewidth=1000, precision=4)
        # print(np.linalg.norm(jac-jac_est))
        if np.linalg.norm(self.__state) < 1e-13:
            print("First solve: use fsolve")
            soln = fsolve(build_residual, self.__state, fprime=build_jacobian, xtol=1e-13)
        else:
            diff = 1
            soln = self.__state.copy()
            count = 0
            while diff > 1e-13:
                update = np.linalg.solve(build_jacobian(soln), build_residual(soln))
                soln -= update
                diff = np.linalg.norm(update)
                count += 1
                if count > 10:
                    print("Newton failed: use fsolve")
                    soln = fsolve(build_residual, self.__state, fprime=build_jacobian, xtol=1e-13)
                    break

        self.__state[:] = soln[:]
        sigma = self.__state[:-1]
        iota = self.__state[-1]
        dsigma_by_dphi = self.D @ sigma
        return sigma, iota, dsigma_by_dphi
    
    def compute_by_dcoefficients(self, order=4):
        ma = self.magnetic_axis
        numpoints = len(ma.points)
        eps = 1e-5
        x0 = ma.get_dofs()
        numcoeffs = len(x0)
        dB_by_dcoeffs = np.zeros((numpoints, numcoeffs, 3))
        d2B_by_dcoeffsdX = np.zeros((numpoints, numcoeffs, 3, 3))
        diota_by_dcoeffs = np.zeros((numcoeffs, 1)) 
        if order == 2:
            for i in range(numcoeffs):
                x = x0.copy()
                x[i] += eps
                ma.set_dofs(x)
                self.clear()
                dB_by_dcoeffs[:, i, :] = self.B
                d2B_by_dcoeffsdX[:, i, :, :] = self.dB_by_dX
                diota_by_dcoeffs[i, 0] = self.iota
                x[i] -= 2*eps
                ma.set_dofs(x)
                self.clear()
                dB_by_dcoeffs[:, i, :] -= self.B
                dB_by_dcoeffs[:, i, :] *= 1/(2*eps)
                d2B_by_dcoeffsdX[:, i, :, :] -= self.dB_by_dX
                d2B_by_dcoeffsdX[:, i, :, :] *= 1/(2*eps)
                diota_by_dcoeffs[i, 0] -= self.iota
                diota_by_dcoeffs[i, 0] *= 1/(2*eps)
        elif order == 4:
            for i in range(numcoeffs):
                x = x0.copy()

                x[i] += 2*eps
                ma.set_dofs(x)
                self.clear()
                a = -1./12
                dB_by_dcoeffs[:, i, :]       = a * self.B
                d2B_by_dcoeffsdX[:, i, :, :] = a * self.dB_by_dX
                diota_by_dcoeffs[i, 0]       = a * self.iota

                x[i] -= eps
                ma.set_dofs(x)
                self.clear()
                a = 2./3
                dB_by_dcoeffs[:, i, :]       += a * self.B
                d2B_by_dcoeffsdX[:, i, :, :] += a * self.dB_by_dX
                diota_by_dcoeffs[i, 0]       += a * self.iota

                x[i] -= 2*eps
                ma.set_dofs(x)
                self.clear()
                a = -2./3
                dB_by_dcoeffs[:, i, :]       += a * self.B
                d2B_by_dcoeffsdX[:, i, :, :] += a * self.dB_by_dX
                diota_by_dcoeffs[i, 0]       += a * self.iota

                x[i] -= eps
                ma.set_dofs(x)
                self.clear()
                a = 1./12
                dB_by_dcoeffs[:, i, :]       += a * self.B
                d2B_by_dcoeffsdX[:, i, :, :] += a * self.dB_by_dX
                diota_by_dcoeffs[i, 0]       += a * self.iota

                dB_by_dcoeffs[:, i, :]       *= 1/(eps)
                d2B_by_dcoeffsdX[:, i, :, :] *= 1/(eps)
                diota_by_dcoeffs[i, 0]       *= 1/(eps)
        else:
            raise NotImplementedError

        ma.set_dofs(x0)
        self.clear()
        self.dB_by_dcoeffs = dB_by_dcoeffs
        self.d2B_by_dcoeffsdX = d2B_by_dcoeffsdX
        self.diota_by_dcoeffs = diota_by_dcoeffs

    @writable_cached_property
    def dB_by_dcoeffs(self):
        self.compute_by_dcoefficients()
        return self.dB_by_dcoeffs
    
    @writable_cached_property
    def d2B_by_dcoeffsdX(self):
        self.compute_by_dcoefficients()
        return self.d2B_by_dcoeffsdX
    
    @writable_cached_property
    def diota_by_dcoeffs(self):
        self.compute_by_dcoefficients()
        return self.diota_by_dcoeffs
    
    def compute_by_detabar(self):
        ma = self.magnetic_axis
        numpoints = len(ma.points)
        eps = 1e-6
        dB_by_detabar = np.zeros((numpoints, 1, 3))
        d2B_by_detabardX = np.zeros((numpoints, 1, 3, 3))
        diota_by_detabar = np.zeros((1, 1)) 
        self.eta_bar += eps
        self.clear()
        dB_by_detabar[:, 0, :] = self.B
        d2B_by_detabardX[:, 0, :, :] = self.dB_by_dX
        diota_by_detabar[0, 0] = self.iota
        self.eta_bar -= 2*eps
        self.clear()
        dB_by_detabar[:, 0, :] -= self.B
        dB_by_detabar[:, 0, :] *= 1/(2*eps)
        d2B_by_detabardX[:, 0, :, :] -= self.dB_by_dX
        d2B_by_detabardX[:, 0, :, :] *= 1/(2*eps)
        diota_by_detabar[0, 0] -= self.iota
        diota_by_detabar[0, 0] *= 1/(2*eps)
        self.eta_bar += eps
        self.clear()
        self.dB_by_detabar = dB_by_detabar
        self.d2B_by_detabardX = d2B_by_detabardX
        self.diota_by_detabar = diota_by_detabar

    @writable_cached_property
    def dB_by_detabar(self):
        self.compute_by_detabar()
        return self.dB_by_detabar

    @writable_cached_property
    def d2B_by_detabardX(self):
        self.compute_by_detabar()
        return self.d2B_by_detabardX

    @writable_cached_property
    def diota_by_detabar(self):
        self.compute_by_detabar()
        return self.diota_by_detabar
