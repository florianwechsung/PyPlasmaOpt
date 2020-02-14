import numpy as np
from math import pi
import cppplasmaopt as cpp
from property_manager3 import cached_property, PropertyManager
writable_cached_property = cached_property(writable=True)


class BiotSavart(PropertyManager):

    def __init__(self, coils, coil_currents):
        assert len(coils) == len(coil_currents)
        self.coils = coils
        self.coil_currents = coil_currents

    def set_points(self, points):
        self.points = points
        self.clear_cached_properties()

    @writable_cached_property
    def B(self):
        self.compute(self.points)
        return self.B

    @writable_cached_property
    def dB_by_dX(self):
        self.compute(self.points)
        return self.dB_by_dX

    @writable_cached_property
    def d2B_by_dXdX(self):
        self.compute(self.points)
        return self.d2B_by_dXdX

    @writable_cached_property
    def dB_by_dcoilcurrents(self):
        self.compute(self.points)
        return self.dB_by_dcoilcurrents

    @writable_cached_property
    def d2B_by_dXdcoilcurrents(self):
        self.compute(self.points)
        return self.d2B_by_dXdcoilcurrents

    @writable_cached_property
    def dB_by_dcoilcoeffs(self):
        self.compute_by_dcoilcoeff(self.points)
        return self.dB_by_dcoilcoeffs

    @writable_cached_property
    def d2B_by_dXdcoilcoeffs(self):
        self.compute_by_dcoilcoeff(self.points)
        return self.d2B_by_dXdcoilcoeffs

    def compute(self, points, use_cpp=True):
        self.B           = np.zeros((len(points), 3))
        self.dB_by_dX    = np.zeros((len(points), 3, 3))
        self.d2B_by_dXdX = np.zeros((len(points), 3, 3, 3))
        self.dB_by_dcoilcurrents    = [np.zeros((len(points), 3)) for coil in self.coils]
        self.d2B_by_dXdcoilcurrents = [np.zeros((len(points), 3, 3)) for coil in self.coils]

        if use_cpp:
            gammas                 = [coil.gamma for coil in self.coils]
            dgamma_by_dphis        = [coil.dgamma_by_dphi[:, 0, :] for coil in self.coils]

            cpp.biot_savart_all(points, gammas, dgamma_by_dphis, self.coil_currents, self.B, self.dB_by_dX, self.d2B_by_dXdX, self.dB_by_dcoilcurrents, self.d2B_by_dXdcoilcurrents)
        else:
            for l in range(len(self.coils)):
                coil = self.coils[l]
                current = self.coil_currents[l]
                gamma = coil.gamma
                dgamma_by_dphi = coil.dgamma_by_dphi[:, 0, :]
                num_coil_quadrature_points = gamma.shape[0]
                for i, point in enumerate(points):
                    diff = point-gamma
                    self.dB_by_dcoilcurrents[l][i, :] += np.sum(
                        (1./np.linalg.norm(diff, axis=1)**3)[:, None] * np.cross(dgamma_by_dphi, diff, axis=1),
                        axis=0)
                self.dB_by_dcoilcurrents[l] *= (1e-7/num_coil_quadrature_points)
                self.B += current * self.dB_by_dcoilcurrents[l]
            for l in range(len(self.coils)):
                coil = self.coils[l]
                current = self.coil_currents[l]
                gamma = coil.gamma
                dgamma_by_dphi = coil.dgamma_by_dphi[:, 0, :]
                num_coil_quadrature_points = gamma.shape[0]
                for i, point in enumerate(points):
                    diff = point-gamma
                    norm_diff = np.linalg.norm(diff, axis=1)
                    dgamma_by_dphi_cross_diff = np.cross(dgamma_by_dphi, diff, axis=1)
                    for j in range(3):
                        ek = np.zeros((3,))
                        ek[j] = 1.
                        numerator1 = norm_diff[:, None] * np.cross(dgamma_by_dphi, ek)
                        numerator2 = (3.*diff[:, j]/norm_diff)[:, None] * dgamma_by_dphi_cross_diff
                        self.d2B_by_dXdcoilcurrents[l][i, j, :] += np.sum((1./norm_diff**4)[:, None]*(numerator1-numerator2),
                            axis=0)
                self.d2B_by_dXdcoilcurrents[l] *= (1e-7/num_coil_quadrature_points)
                self.dB_by_dX += current * self.d2B_by_dXdcoilcurrents[l]
            for coil, current in zip(self.coils, self.coil_currents):
                gamma = coil.gamma
                dgamma_by_dphi = coil.dgamma_by_dphi[:, 0, :]
                num_coil_quadrature_points = gamma.shape[0]
                for i, point in enumerate(points):
                    diff = point-gamma
                    norm_diff = np.linalg.norm(diff, axis=1)
                    dgamma_by_dphi_cross_diff = np.cross(dgamma_by_dphi, diff, axis=1)
                    for j1 in range(3):
                        for j2 in range(3):
                            ej1 = np.zeros((3,))
                            ej2 = np.zeros((3,))
                            ej1[j1] = 1.
                            ej2[j2] = 1.
                            term1 = -3 * (diff[:, j1]/norm_diff**5)[:, None] * np.cross(dgamma_by_dphi, ej2)
                            term2 = -3 * (diff[:, j2]/norm_diff**5)[:, None] * np.cross(dgamma_by_dphi, ej1)
                            term3 = 15 * (diff[:, j1] * diff[:, j2] / norm_diff**7)[:, None] * dgamma_by_dphi_cross_diff
                            if j1 == j2:
                                term4 = -3 * (1./norm_diff**5)[:, None] * dgamma_by_dphi_cross_diff
                            else:
                                term4 = 0
                            self.d2B_by_dXdX[i, j1, j2, :] += (current/num_coil_quadrature_points) * np.sum(term1 + term2 + term3 + term4, axis=0)
            self.d2B_by_dXdX *= 1e-7
        return self

    def compute_by_dcoilcoeff(self, points, use_cpp=True):
        self.dB_by_dcoilcoeffs    = [np.zeros((len(points), coil.dgamma_by_dcoeff.shape[1], 3)) for coil in self.coils]
        self.d2B_by_dXdcoilcoeffs = [np.zeros((len(points), coil.dgamma_by_dcoeff.shape[1], 3, 3)) for coil in self.coils]
        if use_cpp:
            gammas                 = [coil.gamma for coil in self.coils]
            dgamma_by_dphis        = [coil.dgamma_by_dphi[:, 0, :] for coil in self.coils]
            dgamma_by_dcoeffs      = [coil.dgamma_by_dcoeff for coil in self.coils]
            d2gamma_by_dphidcoeffs = [coil.d2gamma_by_dphidcoeff[:, 0, :, :] for coil in self.coils]


            cpp.biot_savart_by_dcoilcoeff_all(points, gammas, dgamma_by_dphis, dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, self.coil_currents, self.dB_by_dcoilcoeffs, self.d2B_by_dXdcoilcoeffs)
        else:
            for l in range(len(self.coils)):
                coil = self.coils[l]
                current = self.coil_currents[l]
                gamma = coil.gamma
                dgamma_by_dphi = coil.dgamma_by_dphi[:, 0, :]
                dgamma_by_dcoeff = coil.dgamma_by_dcoeff
                d2gamma_by_dphidcoeff = coil.d2gamma_by_dphidcoeff[:, 0, :, :]
                num_coil_quadrature_points = gamma.shape[0]
                num_coil_coeffs = dgamma_by_dcoeff.shape[1]
                for i, point in enumerate(points):
                    diff = point-gamma
                    norm_diff = np.linalg.norm(diff, axis=1)
                    norm_diff_3_inv = (1./norm_diff**3)[:, None]
                    norm_diff_5_inv = (1./norm_diff**5)[:, None]
                    dgamma_by_dphi_cross_diff = np.cross(dgamma_by_dphi, diff, axis=1)
                    for j in range(num_coil_coeffs):
                        term1 = norm_diff_3_inv * np.cross(d2gamma_by_dphidcoeff[:, j, :], diff, axis=1)
                        term2 = norm_diff_3_inv * np.cross(dgamma_by_dphi, dgamma_by_dcoeff[:, j, :], axis=1)
                        term3 = norm_diff_5_inv * np.sum(dgamma_by_dcoeff[:, j, :] * diff, axis=1)[:, None] * dgamma_by_dphi_cross_diff * 3
                        self.dB_by_dcoilcoeffs[l][i, j, :] += current * np.sum(term1-term2+term3, axis=0)
                mu = 4 * pi * 1e-7
                self.dB_by_dcoilcoeffs[l] *= mu/(4*pi*num_coil_quadrature_points)
            for l in range(len(self.coils)):
                coil = self.coils[l]
                current = self.coil_currents[l]
                gamma = coil.gamma
                dgamma_by_dphi = coil.dgamma_by_dphi[:, 0, :]
                dgamma_by_dcoeff = coil.dgamma_by_dcoeff
                d2gamma_by_dphidcoeff = coil.d2gamma_by_dphidcoeff[:, 0, :, :]
                num_coil_coeffs = dgamma_by_dcoeff.shape[1]
                num_coil_quadrature_points = gamma.shape[0]
                for i, point in enumerate(points):
                    diff = point-gamma
                    norm_diff = np.linalg.norm(diff, axis=1)
                    dgamma_by_dphi_cross_diff = np.cross(dgamma_by_dphi, diff, axis=1)
                    for j in range(num_coil_coeffs):
                        for k in range(3):
                            ek = np.zeros((3,))
                            ek[k] = 1.
                            term1 = (1/norm_diff**3)[:, None] * np.cross(d2gamma_by_dphidcoeff[:, j, :], ek)
                            term2 = 3 * (np.sum(diff * dgamma_by_dcoeff[:, j, :], axis=1)/norm_diff**5)[:, None] * np.cross(dgamma_by_dphi, ek)
                            term3 = -15 * (np.sum(diff * dgamma_by_dcoeff[:, j, :], axis=1) * diff[:, k]/norm_diff**7)[:, None] * dgamma_by_dphi_cross_diff
                            term4 = 3 * (dgamma_by_dcoeff[:,j,k]/norm_diff**5)[:, None] * dgamma_by_dphi_cross_diff
                            term5 = -3 * (diff[:, k]/norm_diff**5)[:, None] * np.cross(d2gamma_by_dphidcoeff[:, j, :], diff, axis=1)
                            term6 = 3 * (diff[:, k]/norm_diff**5)[:, None] * np.cross(dgamma_by_dphi, dgamma_by_dcoeff[:, j, :])
                            self.d2B_by_dXdcoilcoeffs[l][i, j, k, :] = current * np.sum(term1 + term2 + term3 + term4 + term5 + term6, axis=0)
                mu = 4 * pi * 1e-7
                self.d2B_by_dXdcoilcoeffs[l] *= mu/(4*pi*num_coil_quadrature_points)
        return self
