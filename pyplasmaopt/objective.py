import numpy as np


class SquaredMagneticFieldNormOnCurve(object):

    r"""
    This objective calculates
        J = \int_{curve} |B(s)|^2 ds
    given a curve and a Biot Savart kernel.
    """

    def __init__(self, curve, biotsavart, num_quadrature_points):
        self.curve = curve
        self.biotsavart = biotsavart
        self.num_quadrature_points = num_quadrature_points

    def J(self):
        phis = np.linspace(0, 1, self.num_quadrature_points, endpoint=False)
        quadrature_points = self.curve.gamma(phis)
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi(phis)[:,0,:], axis=1)
        B = self.biotsavart.B(quadrature_points)
        return np.sum(arc_length[:, None] * (B**2))/self.num_quadrature_points

    def dJ_by_dcoilcoefficients(self):
        """
        Calculate the derivatives with respect to the coil coefficients.
        """

        phis = np.linspace(0, 1, self.num_quadrature_points, endpoint=False)
        quadrature_points = self.curve.gamma(phis)
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi(phis)[:,0,:], axis=1)

        B = self.biotsavart.B(quadrature_points)
        dB_by_dcoilcoeff = self.biotsavart.dB_by_dcoilcoeff(quadrature_points)
        res = []
        for dB in dB_by_dcoilcoeff:
            temp = np.zeros((dB.shape[1],))
            for i in range(dB.shape[1]):
                for k in range(3):
                    temp[i] += np.sum(B[:, k] * dB[:, i, k] * arc_length *2 )
            res.append(temp/self.num_quadrature_points)
        return res

    def dJ_by_dcurvecoefficients(self):
        """
        Calculate the derivatives with respect to the coefficients describing
        the shape of the curve that we are integrating over the magnetic field
        over.
        """
        phis = np.linspace(0, 1, self.num_quadrature_points, endpoint=False)

        gamma                 = self.curve.gamma(phis)
        dgamma_by_dphi        = self.curve.dgamma_by_dphi(phis)[:,0,:]
        dgamma_by_dcoeff      = self.curve.dgamma_by_dcoeff(phis)
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff(phis)[:, 0, :, :]

        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)

        B        = self.biotsavart.B(gamma)
        dB_by_dX = self.biotsavart.dB_by_dX(gamma)

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            for k1 in range(3):
                for k2 in range(3):
                    res[i] += 2 * np.sum(B[:, k1] * dB_by_dX[:, k1, k2] * dgamma_by_dcoeff[:, i, k2] * arc_length)
            res[i] += np.sum((1/arc_length) * np.sum(B**2, axis=1) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        res *= 1/self.num_quadrature_points
        return res
