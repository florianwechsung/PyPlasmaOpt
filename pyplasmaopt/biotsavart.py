import numpy as np
from math import pi
import cppplasmaopt as cpp


class BiotSavart():

    def __init__(self, coils, coil_currents, num_coil_quadrature_points):
        assert len(coils) == len(coil_currents)
        self.coils = coils
        self.coil_currents = coil_currents
        self.coil_quadrature_points = np.linspace(0, 1, num_coil_quadrature_points, endpoint=False)
        self.num_coil_quadrature_points = num_coil_quadrature_points

    def B(self, points, use_cpp=True):
        res = np.zeros((len(points), 3))
        for coil, current in zip(self.coils, self.coil_currents):
            gamma = coil.gamma(self.coil_quadrature_points)
            dgamma_by_dphi = coil.dgamma_by_dphi(self.coil_quadrature_points)[:, 0, :]
            if use_cpp:
                res += current * cpp.biot_savart_B(points, gamma, dgamma_by_dphi)
            else:
                for i, point in enumerate(points):
                    diff = point-gamma
                    res[i, :] += current * np.sum(
                        (1./np.linalg.norm(diff, axis=1)**3)[:, None] * np.cross(dgamma_by_dphi, diff, axis=1),
                        axis=0)
        mu = 4 * pi * 1e-7
        res *= mu/(4*pi*self.num_coil_quadrature_points)
        return res
    
    def dB_dX(self, points):
        res = np.zeros((len(points), 3, 3))
        for coil, current in zip(self.coils, self.coil_currents):
            gamma = coil.gamma(self.coil_quadrature_points)
            dgamma_by_dphi = coil.dgamma_by_dphi(self.coil_quadrature_points)[:, 0, :]
            for i, point in enumerate(points):
                diff = point-gamma
                norm_diff = np.linalg.norm(diff, axis=1)
                dgamma_by_dphi_cross_diff = np.cross(dgamma_by_dphi, diff, axis=1)
                for j in range(3):
                    ek = np.zeros((3,))
                    ek[j] = 1.
                    numerator1 = np.linalg.norm(diff, axis=1)[:, None] * np.cross(dgamma_by_dphi, ek)
                    numerator2 = (3.*diff[:, j]/norm_diff)[:, None] * dgamma_by_dphi_cross_diff
                    res[i, :, j] += current * np.sum((1./norm_diff**4)[:, None]*(numerator1-numerator2),
                        axis=0)
        mu = 4 * pi * 1e-7
        res *= mu/(4*pi*self.num_coil_quadrature_points)
        return res

    def dB_dcoil(self, points, coil_direction):
        raise NotImplementedError

    def d2B_dXdcoil(self, points, coil_direction):
        raise NotImplementedError
