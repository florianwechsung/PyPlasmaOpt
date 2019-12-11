from .curve import Curve
import numpy as np
from math import pi
from scipy.optimize import fsolve


class QuasiSymmetricField():

    def __init__(self, eta_bar, magnetic_axis):
        self.s_G = 1
        self.B_0 = 1
        self.s_Psi = 1
        self.eta_bar = eta_bar
        self.magnetic_axis = magnetic_axis
        self.n = len(magnetic_axis.points)
        self.state = np.zeros((self.n+2,))

    def solve_state(self):

        ldash = self.magnetic_axis.incremental_arclength[:, 0]
        kappa = self.magnetic_axis.kappa[:, 0]

        G_0  = np.mean(ldash) * self.s_G * self.B_0/(2*pi)
        fak1 = abs(G_0)/self.B_0
        fak2 = 2 * G_0 * self.eta_bar**2 / (self.s_Psi * self.B_0)

        ldash_padded = np.append(ldash, ldash[0])
        kappa_padded = np.append(kappa, kappa[0])
        torsion = self.magnetic_axis.torsion[:, 0]
        torsion_padded = np.append(torsion, torsion[0])
        h = self.magnetic_axis.points[1] - self.magnetic_axis.points[0]

        def build_residual(x):
            sigma = x[:-1]
            iota = x[-1]
            residual = np.zeros(x.shape)
            residual[0] = x[0]
            for i in range(1, self.n):
                residual[i] = (fak1/ldash_padded[i]) * (sigma[i+1] - sigma[i-1])/(2*h) \
                    + iota * ((self.eta_bar/kappa_padded[i])**4 + 1 + sigma[i]**2) \
                    + fak2 * torsion_padded[i] / kappa_padded[i]**2
            i = self.n
            residual[i] = (fak1/ldash_padded[i]) * (sigma[1] - sigma[i-1])/(2*h) \
                + iota * ((self.eta_bar/kappa_padded[i])**4 + 1 + sigma[i]**2) \
                + fak2 * torsion_padded[i] / kappa_padded[i]**2
            residual[self.n+1] = x[self.n]-x[0]
            return residual

        def build_jacobian(x):
            jacobian = np.zeros((self.n+2, self.n+2))
            sigma = x[:-1]
            iota = x[-1]
            jacobian[0, 0] = 1
            for i in range(1, self.n):
                jacobian[i, i+1] = (fak1/ldash_padded[i])*1/(2*h)
                jacobian[i, i]   = 2 * iota * sigma[i]
                jacobian[i, i-1] = -(fak1/ldash_padded[i])*1/(2*h)
                jacobian[i, self.n+1] = ((self.eta_bar/kappa_padded[i])**4 + 1 + sigma[i]**2)
            i = self.n
            jacobian[i, 1] = (fak1/ldash_padded[i])*1/(2*h)
            jacobian[i, i]   = 2 * iota * sigma[i]
            jacobian[i, i-1] = -(fak1/ldash_padded[i])*1/(2*h)
            jacobian[i, self.n+1] = ((self.eta_bar/kappa_padded[i])**4 + 1 + sigma[i]**2)
            jacobian[self.n+1, self.n] = 1
            jacobian[self.n+1, 0] = -1
            return jacobian

        # x = np.random.rand(*self.state.shape)
        # jac = build_jacobian(x)
        # jac_est = np.zeros(jac.shape)
        # f0 = build_residual(x)
        # eps = 1e-6
        # for i in range(self.n+2):
        #     x[i] += eps
        #     fx = build_residual(x)
        #     x[i] -= eps
        #     jac_est[:, i] = (fx-f0)/eps
        # np.set_printoptions(linewidth=1000, precision=4)
        # # print(np.round(jac_est, 2))
        # # print(np.round(jac, 2))
        # print(np.linalg.norm(jac-jac_est))
        res = fsolve(build_residual, self.state, fprime=build_jacobian, xtol=1e-10)
        self.state[:] = res[:]
        sigma = res[:-2]
        self.dsigma_by_dphi = np.asarray([(sigma[(i+1)%self.n]-sigma[i-1])/(2*h) for i in range(self.n)])
        return (sigma, res[-1])

    def B(self):
        (t, n, b) = self.magnetic_axis.frenet_frame
        return self.B_0 * t

    def dB_by_dX(self):
        (t, n, b) = self.magnetic_axis.frenet_frame
        kappa = self.magnetic_axis.kappa[:,0]
        dkappa_by_dphi = self.magnetic_axis.dkappa_by_dphi[:,0,0]
        torsion = self.magnetic_axis.torsion[:,0]
        ldash = self.magnetic_axis.incremental_arclength[:,0]
        s_Psi = self.s_Psi
        s_G = self.s_G
        B_0 = self.B_0
        G_0 = np.mean(ldash) * self.s_G * self.B_0/(2*pi)
        eta_bar = self.eta_bar
        iota = self.state[-1]
        sigma = self.state[:-2]
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
        res = np.zeros((self.n, 3, 3))
        for j in range(3):
            nterm = s_Psi * G_0 * kappa * t[:, j] / B_0
            nterm += (dX1c_dvarphi * Y1s + iota * X1c * Y1c) * n[:, j]
            nterm += (dY1c_dvarphi * Y1s - dY1s_dvarphi * Y1c + s_Psi * G_0 * B_0 * torsion + iota*(Y1s**2 + Y1c**2)) * b[:, j]
            bterm = (-s_Psi * G_0 * torsion/B_0 - iota * X1c**2) * n[:, j]
            bterm += (X1c * dY1s_dvarphi - iota * X1c * Y1c) * b[:, j]
            tterm = kappa * s_G * B_0 * n[:, j]
            res[:, j, :] = s_Psi * (B_0**2/abs(G_0)) * (nterm[:, None] * n + bterm[:, None] * b) + tterm[:, None] * t
        return res
