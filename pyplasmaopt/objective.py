import numpy as np


class BiotSavartQuasiSymmetricFieldDifference():

    def __init__(self, quasi_symmetric_field, biotsavart):
        self.quasi_symmetric_field = quasi_symmetric_field
        self.biotsavart = biotsavart

    def J_L2(self):
        Bbs        = self.biotsavart.B
        Bqs        = self.quasi_symmetric_field.B
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        return np.sum(arc_length[:, None] * (Bbs-Bqs)**2)/len(arc_length)

    def dJ_L2_by_dcoilcoefficients(self):
        Bbs                = self.biotsavart.B
        Bqs                = self.quasi_symmetric_field.B
        dBbs_by_dcoilcoeff = self.biotsavart.dB_by_dcoilcoeffs
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        for dB in dBbs_by_dcoilcoeff:
            res.append(np.einsum('ij,ikj,i->k', Bbs-Bqs, dB, arc_length) * 2 / len(arc_length))
        return res

    def dJ_L2_by_dcoilcurrents(self):
        Bbs                   = self.biotsavart.B
        dBbs_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents
        Bqs                   = self.quasi_symmetric_field.B
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        for dB in dBbs_by_dcoilcurrents:
            res.append(np.einsum('ij,ij,i', Bbs-Bqs, dB, arc_length) * 2 / len(arc_length))
        return res

    def dJ_L2_by_dmagneticaxiscoefficients(self):
        dgamma_by_dphi        = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.quasi_symmetric_field.magnetic_axis.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]

        Bbs        = self.biotsavart.B
        dBbs_by_dX = self.biotsavart.dB_by_dX
        Bqs        = self.quasi_symmetric_field.B
        dBqs_by_dcoeffs = self.quasi_symmetric_field.dB_by_dcoeffs

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            for k1 in range(3):
                for k2 in range(3):
                    res[i] += 2 * np.sum((Bbs[:, k1]-Bqs[:, k1]) * dBbs_by_dX[:, k1, k2] * dgamma_by_dcoeff[:, i, k2] * arc_length)
            res[i] -= np.sum(2*(Bbs-Bqs)*dBqs_by_dcoeffs[:, i, :] * arc_length[:, None])
            res[i] += np.sum((1/arc_length) * np.sum((Bbs-Bqs)**2, axis=1) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        res *= 1/arc_length.shape[0]
        return res

    def dJ_L2_by_detabar(self):
        Bbs             = self.biotsavart.B
        arc_length      = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        Bqs             = self.quasi_symmetric_field.B
        dBqs_by_detabar = self.quasi_symmetric_field.dB_by_detabar
        res = np.zeros((1, ))
        res[0] -= np.sum(2*(Bbs-Bqs)*dBqs_by_detabar[:, 0, :] * arc_length[:, None])
        res *= 1/arc_length.shape[0]
        return res

    def J_H1(self):
        dBbs_by_dX = self.biotsavart.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        dBqs_by_dX = self.quasi_symmetric_field.dB_by_dX
        return np.sum(arc_length[:, None, None] * (dBbs_by_dX-dBqs_by_dX)**2)/len(arc_length)

    def dJ_H1_by_dcoilcoefficients(self):
        dBbs_by_dX            = self.biotsavart.dB_by_dX
        d2Bbs_by_dXdcoilcoeff = self.biotsavart.d2B_by_dXdcoilcoeffs
        dBqs_by_dX            = self.quasi_symmetric_field.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        for dB in d2Bbs_by_dXdcoilcoeff:
            res.append(np.einsum('ijk,iljk,i->l', dBbs_by_dX-dBqs_by_dX, dB, arc_length) * 2 / len(arc_length))
        return res

    def dJ_H1_by_dcoilcurrents(self):
        dBbs_by_dX               = self.biotsavart.dB_by_dX
        d2Bbs_by_dXdcoilcurrents = self.biotsavart.d2B_by_dXdcoilcurrents
        dBqs_by_dX               = self.quasi_symmetric_field.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        for dB in d2Bbs_by_dXdcoilcurrents:
            res.append(np.einsum('ijk,ijk,i', dBbs_by_dX-dBqs_by_dX, dB, arc_length) * 2 / len(arc_length))
        return res

    def dJ_H1_by_dmagneticaxiscoefficients(self):

        gamma                 = self.quasi_symmetric_field.magnetic_axis.gamma
        dgamma_by_dphi        = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.quasi_symmetric_field.magnetic_axis.d2gamma_by_dphidcoeff[:, 0, :, :]
        d2Bqs_by_dcoeffsdX    = self.quasi_symmetric_field.d2B_by_dcoeffsdX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]

        dBbs_by_dX    = self.biotsavart.dB_by_dX
        d2Bbs_by_dXdX = self.biotsavart.d2B_by_dXdX
        dBqs_by_dX    = self.quasi_symmetric_field.dB_by_dX

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            for k1 in range(3):
                for k2 in range(3):
                    for k3 in range(3):
                        res[i] += 2.0 * np.sum((dBbs_by_dX[:, k1, k2]-dBqs_by_dX[:, k1, k2]) * d2Bbs_by_dXdX[:, k1, k2, k3] * dgamma_by_dcoeff[:, i, k3] * arc_length)
            res[i] -= np.sum(2*(dBbs_by_dX-dBqs_by_dX)*d2Bqs_by_dcoeffsdX[:, i, :, :] * arc_length[:, None, None])
            res[i] += np.sum((1/arc_length) * np.sum(np.sum((dBbs_by_dX-dBqs_by_dX)**2, axis=1), axis=1) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        res *= 1/gamma.shape[0]
        return res

    def dJ_H1_by_detabar(self):
        dBbs_by_dX         = self.biotsavart.dB_by_dX
        dBqs_by_dX         = self.quasi_symmetric_field.dB_by_dX
        d2Bqs_by_detabardX = self.quasi_symmetric_field.d2B_by_detabardX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]

        res = np.zeros((1, ))
        res[0] -= np.sum(2*(dBbs_by_dX-dBqs_by_dX)*d2Bqs_by_detabardX[:, 0, :, :] * arc_length[:, None, None])
        res *= 1/arc_length.shape[0]
        return res


class SquaredMagneticFieldNormOnCurve(object):

    r"""
    This objective calculates
        J = \int_{curve} |B(s)|^2 ds
    given a curve and a Biot Savart kernel.
    """

    def __init__(self, curve, biotsavart):
        self.curve = curve
        self.biotsavart = biotsavart

    def J(self):
        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        B = self.biotsavart.compute(quadrature_points).B
        return np.sum(arc_length[:, None] * (B**2))/quadrature_points.shape[0]

    def dJ_by_dcoilcoefficients(self):
        """
        Calculate the derivatives with respect to the coil coefficients.
        """

        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)

        B = self.biotsavart.compute(quadrature_points).B
        dB_by_dcoilcoeff = self.biotsavart.compute_by_dcoilcoeff(quadrature_points).dB_by_dcoilcoeffs
        res = []
        for dB in dB_by_dcoilcoeff:
            res.append(np.einsum('ij,ikj,i->k', B, dB, arc_length) * 2 / quadrature_points.shape[0])
        return res

    def dJ_by_dcurvecoefficients(self):
        """
        Calculate the derivatives with respect to the coefficients describing
        the shape of the curve that we are integrating the magnetic field over.
        """

        gamma                 = self.curve.gamma
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.curve.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]

        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        self.biotsavart.compute(gamma)
        B        = self.biotsavart.B
        dB_by_dX = self.biotsavart.dB_by_dX

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            for k1 in range(3):
                for k2 in range(3):
                    res[i] += 2 * np.sum(B[:, k1] * dB_by_dX[:, k1, k2] * dgamma_by_dcoeff[:, i, k2] * arc_length)
            res[i] += np.sum((1/arc_length) * np.sum(B**2, axis=1) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        res *= 1/gamma.shape[0]
        return res

class SquaredMagneticFieldGradientNormOnCurve(object):

    r"""
    This objective calculates
        J = \int_{curve} |∇B(s)|^2 ds
    given a curve and a Biot Savart kernel.
    """

    def __init__(self, curve, biotsavart):
        self.curve = curve
        self.biotsavart = biotsavart

    def J(self):
        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        dB_by_dX = self.biotsavart.compute(quadrature_points).dB_by_dX
        return np.sum(arc_length * (np.sum(np.sum(dB_by_dX**2, axis=1), axis=1)))/quadrature_points.shape[0]

    def dJ_by_dcoilcoefficients(self):
        """
        Calculate the derivatives with respect to the coil coefficients.
        """

        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)

        dB_by_dX = self.biotsavart.compute(quadrature_points).dB_by_dX
        d2B_by_dXdcoilcoeff = self.biotsavart.compute_by_dcoilcoeff(quadrature_points).d2B_by_dXdcoilcoeffs
        res = []
        for dB in d2B_by_dXdcoilcoeff:
            res.append(np.einsum('ijk,iljk,i->l', dB_by_dX, dB, arc_length) * 2 / quadrature_points.shape[0])
        return res

    def dJ_by_dcurvecoefficients(self):
        """
        Calculate the derivatives with respect to the coefficients describing
        the shape of the curve that we are integrating the gradient of the
        magnetic field over.
        """

        gamma                 = self.curve.gamma
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.curve.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]

        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        self.biotsavart.compute(gamma)
        dB_by_dX = self.biotsavart.dB_by_dX
        d2B_by_dXdX = self.biotsavart.d2B_by_dXdX

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            for k1 in range(3):
                for k2 in range(3):
                    for k3 in range(3):
                        res[i] += 2.0 * np.sum(dB_by_dX[:, k1, k2] * d2B_by_dXdX[:, k1, k2, k3] * dgamma_by_dcoeff[:, i, k3] * arc_length)
            res[i] += np.sum((1/arc_length) * np.sum(np.sum(dB_by_dX**2, axis=1), axis=1) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        res *= 1/gamma.shape[0]
        return res


class CurveLength():

    r"""
    J = \int_{curve} 1 ds
    """

    def __init__(self, curve):
        self.curve = curve

    def J(self):
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        return np.mean(arc_length)

    def dJ_by_dcoefficients(self):
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        for i in range(num_coeff):
            res[i] = np.mean((1/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        return res


class CurveCurvature():

    r"""
    J = \int_{curve} \kappa ds
    """

    def __init__(self, curve):
        self.curve = curve

    def J(self):
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        kappa = self.curve.kappa[:, 0]
        return np.mean(kappa**2 * arc_length)

    def dJ_by_dcoefficients(self):
        kappa                 = self.curve.kappa[:,0]
        dkappa_by_dcoeff      = self.curve.dkappa_by_dcoeff[:,:,0]
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length            = np.linalg.norm(dgamma_by_dphi, axis=1)

        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res       = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i]  = np.mean((kappa**2/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis  = 1))
            res[i] += np.mean(2*kappa * dkappa_by_dcoeff[:,i] * arc_length)
        return res


class CurveTorsion():

    r"""
    J = \int_{curve} \tau ds
    """

    def __init__(self, curve):
        self.curve = curve

    def J(self):
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        torsion    = self.curve.torsion[:, 0]
        return np.mean(torsion**2 * arc_length)

    def dJ_by_dcoefficients(self):
        torsion               = self.curve.torsion[:,0]
        dtorsion_by_dcoeff    = self.curve.dtorsion_by_dcoeff[:,:,0]
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length            = np.linalg.norm(dgamma_by_dphi, axis=1)

        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res       = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i]  = np.mean((torsion**2/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis  = 1))
            res[i] += np.mean(2*torsion * dtorsion_by_dcoeff[:,i] * arc_length)
        return res
