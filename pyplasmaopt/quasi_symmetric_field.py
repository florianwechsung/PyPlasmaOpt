from .curve import Curve
from .logging import info, warning
import numpy as np
from math import pi
from scipy.optimize import fsolve
from property_manager import cached_property, PropertyManager
writable_cached_property = cached_property(writable=True)


class QuasiSymmetricField(PropertyManager):

    def __init__(self, eta_bar, magnetic_axis):
        self.s_G = 1
        self.B_0 = 1
        self.s_Psi = 1
        self.eta_bar = eta_bar
        self.magnetic_axis = magnetic_axis
        self.n = len(magnetic_axis.quadpoints)
        self.__state = np.zeros((self.n+1,))
        import scipy
        n = self.n
        points = np.asarray(self.magnetic_axis.quadpoints).reshape((n, 1))
        oneton = np.asarray(range(0, n)).reshape((n, 1))
        fak = (2 * pi) / (points[-1] - points[0] + (points[1]-points[0]))
        dists = fak * scipy.spatial.distance.cdist(points, points, lambda a, b: a-b)
        np.fill_diagonal(dists, 1e-10)  # to shut up the warning
        if n % 2 == 0:
            D = 0.5 \
                * np.power(-1, scipy.spatial.distance.cdist(oneton, -oneton)) \
                / np.tan(0.5 * dists)
        else:
            D = 0.5 \
                * np.power(-1, scipy.spatial.distance.cdist(oneton, -oneton)) \
                / np.sin(0.5 * dists)

        np.fill_diagonal(D, 0)
        D *= fak
        self.D = D

    def clear(self):
        self.clear_cached_properties()

    @writable_cached_property
    def B(self):
        self.compute()
        return self.B

    @writable_cached_property
    def dB_by_dX(self):
        self.compute()
        return self.dB_by_dX

    @writable_cached_property
    def sigma(self):
        self.compute()
        return self.sigma

    @writable_cached_property
    def iota(self):
        self.compute()
        return self.iota

    @writable_cached_property
    def dsigma_by_dphi(self):
        self.compute()
        return self.dsigma_by_dphi

    @writable_cached_property
    def diota_by_detabar(self):
        self.compute_derivative()
        return self.diota_by_detabar

    @writable_cached_property
    def dsigma_by_detabar(self):
        self.compute_derivative()
        return self.dsigma_by_detabar

    @writable_cached_property
    def diota_by_dcoeffs(self):
        self.compute_derivative()
        return self.diota_by_dcoeffs

    @writable_cached_property
    def dsigma_by_dcoeffs(self):
        self.compute_derivative()
        return self.dsigma_by_dcoeffs

    @writable_cached_property
    def dB_by_dcoeffs(self):
        self.compute_derivative()
        return self.dB_by_dcoeffs

    @writable_cached_property
    def d2B_by_dcoeffsdX(self):
        self.compute_derivative()
        return self.d2B_by_dcoeffsdX

    @writable_cached_property
    def diota_by_dcoeffs(self):
        self.compute_derivative()
        return self.diota_by_dcoeffs

    @writable_cached_property
    def dB_by_detabar(self):
        self.compute_derivative()
        return self.dB_by_detabar

    @writable_cached_property
    def d2B_by_detabardX(self):
        self.compute_derivative()
        return self.d2B_by_detabardX

    @writable_cached_property
    def diota_by_detabar(self):
        self.compute_derivative()
        return self.diota_by_detabar

    def compute(self):
        sigma, iota, dsigma_by_dphi = self.solve_state()
        self.sigma = sigma
        self.iota = iota
        self.dsigma_by_dphi = dsigma_by_dphi
        """ Compute B """
        (t, n, b) = self.magnetic_axis.frenet_frame()
        self.B = self.B_0 * t

        """ Compute dB_by_dX """
        kappa = self.magnetic_axis.kappa()
        dkappa_by_dphi = self.magnetic_axis.kappadash()
        torsion = self.magnetic_axis.torsion()
        l = self.magnetic_axis.incremental_arclength()
        s_Psi = self.s_Psi
        s_G = self.s_G
        B_0 = self.B_0
        G_0 = np.mean(l) * self.s_G * self.B_0/(2*pi)
        assert G_0 >= 0
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
        dX1c_dvarphi = abs(G_0) * dX1c_dphi/(l * B_0)
        dY1s_dvarphi = abs(G_0) * dY1s_dphi/(l * B_0)
        dY1c_dvarphi = abs(G_0) * dY1c_dphi/(l * B_0)

        self.X1c = X1c
        self.Y1s = Y1s
        self.Y1c = Y1c
        self.dX1c_dphi = dX1c_dphi
        self.dY1s_dphi = dY1s_dphi
        self.dY1c_dphi = dY1c_dphi
        self.dX1c_dvarphi = dX1c_dvarphi
        self.dY1s_dvarphi = dY1s_dvarphi
        self.dY1c_dvarphi = dY1c_dvarphi

        self.dB_by_dX = np.zeros((self.n, 3, 3))
        for j in range(3):
            nterm = s_Psi * G_0 * kappa * t[:, j] / B_0
            nterm += (dX1c_dvarphi * Y1s + iota * X1c * Y1c) * n[:, j]
            nterm += (dY1c_dvarphi * Y1s - dY1s_dvarphi * Y1c + s_Psi * G_0 * B_0 * torsion + iota*(Y1s**2 + Y1c**2)) * b[:, j]
            bterm = (-s_Psi * G_0 * torsion/B_0 - iota * X1c**2) * n[:, j]
            bterm += (X1c * dY1s_dvarphi - iota * X1c * Y1c) * b[:, j]
            tterm = kappa * s_G * B_0 * n[:, j]
            self.dB_by_dX[:, j, :] = s_Psi * (B_0**2/abs(G_0)) * (nterm[:, None] * n + bterm[:, None] * b) + tterm[:, None] * t

    def compute_derivative(self):
        self.compute()
        kappa = self.magnetic_axis.kappa()
        torsion = self.magnetic_axis.torsion()
        (t, n, b) = self.magnetic_axis.frenet_frame()
        l = self.magnetic_axis.incremental_arclength()
        s_G = self.s_G
        s_Psi = self.s_Psi
        eta_bar = self.eta_bar
        B_0 = self.B_0
        G_0 = np.mean(l) * self.s_G * self.B_0/(2*pi)
        assert G_0 >= 0

        iota = self.iota
        sigma = self.sigma
        iota = self.iota
        dsigma_dphi = self.dsigma_by_dphi

        dkappa_by_dphi = self.magnetic_axis.kappadash()

        X1c = self.X1c
        Y1s = self.Y1s
        Y1c = self.Y1c
        dX1c_dphi = self.dX1c_dphi
        dY1s_dphi = self.dY1s_dphi
        dY1c_dphi = self.dY1c_dphi
        dX1c_dvarphi = self.dX1c_dvarphi
        dY1s_dvarphi = self.dY1s_dvarphi
        dY1c_dvarphi = self.dY1c_dvarphi

        (dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff) = self.magnetic_axis.dfrenet_frame_by_dcoeff()
        self.dB_by_detabar = np.zeros((self.B.shape[0], 1, self.B.shape[1]))
        self.dB_by_dcoeffs = self.B_0 * dt_by_dcoeff
        """ Compute d2B_by_detabardX"""

        diota_detabar = self.diota_by_detabar[0, 0]
        dsigma_detabar = self.dsigma_by_detabar[:, 0]
        dX1c_by_detabar = 1/kappa
        dY1s_by_detabar = -s_G * s_Psi * kappa / eta_bar**2
        dY1c_by_detabar = s_G * s_Psi * kappa * dsigma_detabar / eta_bar \
            - s_G * s_Psi * kappa * sigma / eta_bar**2

        d2X1c_dphidetabar = -dkappa_by_dphi / kappa**2
        d2Y1s_dphidetabar = -s_G * s_Psi * dkappa_by_dphi / eta_bar**2
        d2Y1c_dphidetabar = s_G * s_Psi * (dkappa_by_dphi * dsigma_detabar + kappa * self.d2sigma_by_detabardphi[:, 0, 0]) / eta_bar \
            -s_G * s_Psi * (dkappa_by_dphi * sigma + kappa * dsigma_dphi) / eta_bar**2

        d2X1c_dvarphidetabar = abs(G_0) * d2X1c_dphidetabar/(l * B_0)
        d2Y1s_dvarphidetabar = abs(G_0) * d2Y1s_dphidetabar/(l * B_0)
        d2Y1c_dvarphidetabar = abs(G_0) * d2Y1c_dphidetabar/(l * B_0)
        self.d2B_by_detabardX = np.zeros((self.n, 1, 3, 3))
        for j in range(3):
            nterm = (d2X1c_dvarphidetabar * Y1s) * n[:, j]
            nterm += (dX1c_dvarphi * dY1s_by_detabar) * n[:, j]
            nterm += (diota_detabar * X1c * Y1c) * n[:, j]
            nterm += (iota * dX1c_by_detabar * Y1c) * n[:, j]
            nterm += (iota * X1c * dY1c_by_detabar) * n[:, j]

            nterm += (d2Y1c_dvarphidetabar * Y1s - d2Y1s_dvarphidetabar * Y1c + diota_detabar*(Y1s**2 + Y1c**2)) * b[:, j]
            nterm += (dY1c_dvarphi * dY1s_by_detabar - dY1s_dvarphi * dY1c_by_detabar + iota*(2*Y1s*dY1s_by_detabar + 2*Y1c*dY1c_by_detabar)) * b[:, j]

            bterm = (-diota_detabar*X1c**2) * n[:, j]
            bterm += (-iota*2*X1c*dX1c_by_detabar) * n[:, j]
            bterm += (dX1c_by_detabar * dY1s_dvarphi - diota_detabar * X1c * Y1c) * b[:, j]
            bterm += (X1c * d2Y1s_dvarphidetabar - iota * dX1c_by_detabar * Y1c) * b[:, j]
            bterm += (-iota * X1c * dY1c_by_detabar) * b[:, j]
            self.d2B_by_detabardX[:, 0, j, :] = s_Psi * (B_0**2/abs(G_0)) * (nterm[:, None] * n + bterm[:, None] * b)

        diota_by_dcoeffs = self.diota_by_dcoeffs
        dsigma_by_dcoeffs = self.dsigma_by_dcoeffs
        dkappa_by_dcoeff = self.magnetic_axis.dkappa_by_dcoeff()
        dl_by_dcoeff = self.magnetic_axis.dincremental_arclength_by_dcoeff()
        d2sigma_by_dphidcoeff = self.D @ dsigma_by_dcoeffs
        dtorsion_by_dcoeff = self.magnetic_axis.dtorsion_by_dcoeff()

        """ Compute d2B_by_dcoeffsdX"""
        num_coeff = diota_by_dcoeffs.shape[0]
        self.d2B_by_dcoeffsdX = np.zeros((self.n, 3, 3, num_coeff))
        ma = self.magnetic_axis
        d2kappa_by_dphidcoeff = ma.dkappadash_by_dcoeff()
        for i in range(num_coeff):
            dX1c_by_dcoeff = -eta_bar*dkappa_by_dcoeff[:, i]/kappa**2
            dY1s_by_dcoeff = s_G * s_Psi * dkappa_by_dcoeff[:, i] / eta_bar
            dY1c_by_dcoeff = s_G * s_Psi * (dkappa_by_dcoeff[:, i] * sigma + kappa * dsigma_by_dcoeffs[:, i]) / eta_bar

            d2X1c_dphidcoeff = -eta_bar * (d2kappa_by_dphidcoeff[:, i]/kappa**2 - 2 * dkappa_by_dphi * dkappa_by_dcoeff[:, i] / kappa**3)
            d2Y1s_dphidcoeff = s_G * s_Psi * d2kappa_by_dphidcoeff[:, i] / eta_bar
            d2Y1c_dphidcoeff = s_G * s_Psi * (d2kappa_by_dphidcoeff[:, i] * sigma + dkappa_by_dcoeff[:, i] * dsigma_dphi + dkappa_by_dphi * dsigma_by_dcoeffs[:, i] + kappa * d2sigma_by_dphidcoeff[:, i]) / eta_bar

            dG_0_by_dcoeff = np.mean(dl_by_dcoeff[:, i]) * self.s_G * self.B_0/(2*pi)
            d2X1c_dvarphidcoeff = (dG_0_by_dcoeff / B_0) * (dX1c_dphi/l) + (abs(G_0)/B_0) * (d2X1c_dphidcoeff/l-dX1c_dphi*dl_by_dcoeff[:, i]/l**2)
            d2Y1s_dvarphidcoeff = (dG_0_by_dcoeff / B_0) * (dY1s_dphi/l) + (abs(G_0)/B_0) * (d2Y1s_dphidcoeff/l-dY1s_dphi*dl_by_dcoeff[:, i]/l**2)
            d2Y1c_dvarphidcoeff = (dG_0_by_dcoeff / B_0) * (dY1c_dphi/l) + (abs(G_0)/B_0) * (d2Y1c_dphidcoeff/l-dY1c_dphi*dl_by_dcoeff[:, i]/l**2)
            for j in range(3):
                dnterm_by_dcoeff = (s_Psi/B_0) * (dG_0_by_dcoeff * kappa * t[:, j] + G_0 * dkappa_by_dcoeff[:, i] * t[:, j] + G_0 * kappa * dt_by_dcoeff[:, j, i])
                dnterm_by_dcoeff += d2X1c_dvarphidcoeff * Y1s * n[:, j] + dX1c_dvarphi * dY1s_by_dcoeff * n[:, j] + dX1c_dvarphi * Y1s * dn_by_dcoeff[:, j, i]
                dnterm_by_dcoeff += diota_by_dcoeffs[i] * X1c * Y1c * n[:, j] + iota * dX1c_by_dcoeff * Y1c * n[:, j] + iota * X1c * dY1c_by_dcoeff * n[:, j] + iota * X1c * Y1c * dn_by_dcoeff[:, j, i]
                dnterm_by_dcoeff += d2Y1c_dvarphidcoeff * Y1s * b[:, j] + dY1c_dvarphi * dY1s_by_dcoeff * b[:, j] + dY1c_dvarphi * Y1s * db_by_dcoeff[:, j, i]
                dnterm_by_dcoeff -= d2Y1s_dvarphidcoeff * Y1c * b[:, j] + dY1s_dvarphi * dY1c_by_dcoeff * b[:, j] + dY1s_dvarphi * Y1c * db_by_dcoeff[:, j, i]
                dnterm_by_dcoeff += s_Psi * B_0 * (dG_0_by_dcoeff * torsion * b[:, j] + G_0 * dtorsion_by_dcoeff[:, i] * b[:, j] + G_0 * torsion * db_by_dcoeff[:, j, i])
                dnterm_by_dcoeff += diota_by_dcoeffs[i]*(Y1s**2 + Y1c**2) * b[:, j] + iota*(2*dY1s_by_dcoeff * Y1s + 2*Y1c*dY1c_by_dcoeff) * b[:, j] + iota*(Y1s**2 + Y1c**2) * db_by_dcoeff[:, j, i]

                nterm = s_Psi * G_0 * kappa * t[:, j] / B_0
                nterm += (dX1c_dvarphi * Y1s + iota * X1c * Y1c) * n[:, j]
                nterm += (dY1c_dvarphi * Y1s - dY1s_dvarphi * Y1c + s_Psi * G_0 * B_0 * torsion + iota*(Y1s**2 + Y1c**2)) * b[:, j]

                dbterm_by_dcoeff = -(s_Psi/B_0) * (dG_0_by_dcoeff * torsion * n[:, j] + G_0 * dtorsion_by_dcoeff[:, i] * n[:, j] + G_0 * torsion * dn_by_dcoeff[:, j, i])
                dbterm_by_dcoeff -= diota_by_dcoeffs[i] * X1c**2 * n[:, j] + iota * 2 * dX1c_by_dcoeff * X1c * n[:, j] + iota * X1c**2 * dn_by_dcoeff[:, j, i]
                dbterm_by_dcoeff += dX1c_by_dcoeff * dY1s_dvarphi * b[:, j] + X1c * d2Y1s_dvarphidcoeff * b[:, j] + X1c * dY1s_dvarphi * db_by_dcoeff[:, j, i]
                dbterm_by_dcoeff -= diota_by_dcoeffs[i] * X1c * Y1c * b[:, j] + iota * dX1c_by_dcoeff * Y1c * b[:, j] + iota * X1c * dY1c_by_dcoeff * b[:, j] + iota * X1c * Y1c * db_by_dcoeff[:, j, i]

                bterm = (-s_Psi * G_0 * torsion/B_0 - iota * X1c**2) * n[:, j]
                bterm += (X1c * dY1s_dvarphi - iota * X1c * Y1c) * b[:, j]

                dtterm_by_dcoeff = s_G * B_0 * (dkappa_by_dcoeff[:, i] * n[:, j] + kappa  * dn_by_dcoeff[:, j, i])
                tterm = kappa * s_G * B_0 * n[:, j]
                self.d2B_by_dcoeffsdX[:, j, :, i] = -s_Psi * B_0**2 * dG_0_by_dcoeff/G_0**2 * (nterm[:, None] * n + bterm[:, None] * b) \
                    + s_Psi * (B_0**2/abs(G_0)) * (
                          dnterm_by_dcoeff[:, None] * n + nterm[:, None] * dn_by_dcoeff[:, :, i]
                        + dbterm_by_dcoeff[:, None] * b + bterm[:, None] * db_by_dcoeff[:, :, i]
                    ) \
                    + dtterm_by_dcoeff[:, None] * t + tterm[:, None] * dt_by_dcoeff[:, :, i]

    # FIXME: this function could be split up so that the sensitivity is only computed when actually necessary
    def solve_state(self):
        n = self.n
        l = self.magnetic_axis.incremental_arclength()
        kappa = self.magnetic_axis.kappa()

        G_0  = np.mean(l) * self.s_G * self.B_0/(2*pi)
        fak1 = abs(G_0)/self.B_0
        fak2 = 2 * G_0 * self.eta_bar**2 / (self.s_Psi * self.B_0)
        torsion = self.magnetic_axis.torsion()
        fak1lD = np.diag(fak1/l) @ self.D

        def build_residual(x):
            sigma = x[:-1]
            iota = x[-1]
            residual = np.zeros((n+1, ))
            residual[:n] = fak1lD@sigma + iota * ((self.eta_bar/kappa)**4 + 1 + sigma**2) + fak2 * torsion / kappa**2
            residual[-1] = sigma[0]
            return residual

        def build_jacobian(x):
            sigma = x[:-1]
            iota = x[-1]
            jacobian = np.zeros((n+1, n+1))
            jacobian[:n, :n] = fak1lD
            np.fill_diagonal(jacobian[:n, :n], np.diagonal(jacobian[:n, :n]) + 2 * sigma * iota)
            jacobian[:n, n] = ((self.eta_bar/kappa)**4 + 1 + sigma**2)
            jacobian[-1, 0] = 1
            return jacobian

        # x = np.random.rand(*self.__state.shape)
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
        # print( "Jac - Jac_Est", np.linalg.norm(jac-jac_est))
        if np.linalg.norm(self.__state) < 1e-13:
            # info("First solve: use fsolve")
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
                    # warning("Newton failed: use fsolve")
                    soln, _, ier, _ = fsolve(build_residual, self.__state, fprime=build_jacobian, xtol=1e-13, full_output=True)
                    info("ier %s" % ier)
                    if ier != 1:
                        raise RuntimeError("fsolve failed")
                    break

        self.__state[:] = soln[:]
        sigma = self.__state[:-1].copy()
        iota = self.__state[-1]
        dsigma_by_dphi = self.D @ sigma

        jac = build_jacobian(self.__state)
        jacinv = np.linalg.inv(jac)
        """ Calculate dresidual_by_detabar """
        dresidual_by_detabar = np.zeros((n+1, 1))
        dresidual_by_detabar[:n, 0] = iota * 4 * self.eta_bar**3 / kappa**4 + (4 * G_0 * self.eta_bar / (self.s_Psi * self.B_0)) * torsion / kappa**2


        temp = jacinv @ dresidual_by_detabar

        self.dsigma_by_detabar = -temp[:n, :]
        self.diota_by_detabar = -temp[-1:, :]
        self.d2sigma_by_detabardphi = (self.D @ self.dsigma_by_detabar[:, 0]).reshape((n, 1, 1))

        """ Calculate dresidual_by_dcoeffs """
        dl_by_dcoeff = self.magnetic_axis.dincremental_arclength_by_dcoeff()
        dkappa_by_dcoeff = self.magnetic_axis.dkappa_by_dcoeff()

        dG_0_by_dcoeff  = np.mean(dl_by_dcoeff, axis=0) * self.s_G * self.B_0/(2*pi)
        assert G_0 >= 0
        dfak1_by_dcoeff = dG_0_by_dcoeff/self.B_0
        dfak2_by_dcoeff = 2 * dG_0_by_dcoeff * self.eta_bar**2 / (self.s_Psi * self.B_0)
        dtorsion_by_dcoeff = self.magnetic_axis.dtorsion_by_dcoeff()

        num_coeff = dtorsion_by_dcoeff.shape[1]
        dresidual_by_dcoeff = np.zeros((n+1, num_coeff))
        for i in range(num_coeff):
            dresidual_by_dcoeff[:n, i] = (dfak1_by_dcoeff[i]/l - fak1 * dl_by_dcoeff[:,i]/l**2)*(self.D@sigma) \
                -4*iota * dkappa_by_dcoeff[:, i] * (self.eta_bar**4/kappa**5) \
                + dfak2_by_dcoeff[i] * torsion / kappa**2 + fak2 * dtorsion_by_dcoeff[:,i] / kappa**2 - 2*fak2*torsion*dkappa_by_dcoeff[:,i] / kappa**3

        temp = jacinv @ dresidual_by_dcoeff

        self.diota_by_dcoeffs  = -temp[-1:, :].T
        self.dsigma_by_dcoeffs = -temp[:n, :]
        return sigma, iota, dsigma_by_dphi

