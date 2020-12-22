import numpy as np
from math import pi
from property_manager import cached_property, PropertyManager
writable_cached_property = cached_property(writable=True)

class BiotSavartQuasiSymmetricFieldDifference(PropertyManager):

    def __init__(self, quasi_symmetric_field, biotsavart, value_only=False):
        self.quasi_symmetric_field = quasi_symmetric_field
        self.biotsavart = biotsavart
        self.value_only = value_only

    def J_L2(self):
        Bbs        = self.biotsavart.B(compute_derivatives=(1 if self.value_only else 2))
        Bqs        = self.quasi_symmetric_field.B
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength()
        return np.sum(arc_length[:, None] * (Bbs-Bqs)**2)/len(arc_length)


    def compute_dcoilcoefficients(self, reverse_mode=True):
        Bbs        = self.biotsavart.B(compute_derivatives=2)
        Bqs        = self.quasi_symmetric_field.B
        dBbs_by_dX = self.biotsavart.dB_by_dX()
        dBqs_by_dX = self.quasi_symmetric_field.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength()

        v = (Bbs-Bqs) * arc_length[:, None] * 2 / len(arc_length)
        vgrad = (arc_length[:, None, None])*(dBbs_by_dX-dBqs_by_dX) * 2 / len(arc_length)
        if reverse_mode:
            res = self.biotsavart.B_and_dB_vjp(v, vgrad)
            self.dJ_L2_by_dcoilcoefficients = res[0]
            self.dJ_H1_by_dcoilcoefficients = res[1]
        else:
            raise NotImplementedError

    @writable_cached_property
    def dJ_L2_by_dcoilcoefficients(self):
        self.compute_dcoilcoefficients(reverse_mode=True)
        return self.dJ_L2_by_dcoilcoefficients

    def dJ_L2_by_dcoilcurrents(self):
        Bbs                   = self.biotsavart.B(compute_derivatives=2)
        dBbs_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents()
        Bqs                   = self.quasi_symmetric_field.B
        arc_length            = self.quasi_symmetric_field.magnetic_axis.incremental_arclength()
        res = []
        tmp = (Bbs-Bqs) * (arc_length[:, None]*2/len(arc_length))
        return np.einsum('ik,mik->m', tmp, np.asarray(dBbs_by_dcoilcurrents))

    def dJ_L2_by_dmagneticaxiscoefficients(self):
        gammadash            = self.quasi_symmetric_field.magnetic_axis.gammadash()
        dgamma_by_dcoeff     = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dcoeff()
        dgammadash_by_dcoeff = self.quasi_symmetric_field.magnetic_axis.dgammadash_by_dcoeff()
        arc_length           = self.quasi_symmetric_field.magnetic_axis.incremental_arclength()

        Bbs                  = self.biotsavart.B(compute_derivatives=2)
        dBbs_by_dX           = self.biotsavart.dB_by_dX()
        Bqs                  = self.quasi_symmetric_field.B
        dBqs_by_dcoeffs      = self.quasi_symmetric_field.dB_by_dcoeffs
        diff = Bbs-Bqs

        num_coeff = dgamma_by_dcoeff.shape[1]
        # res = 2*np.einsum('ij,ikj,ikm,i->m', diff, dBbs_by_dX, dgamma_by_dcoeff, arc_length)
        # res -= 2*np.einsum('ij,ijm,i->m', diff, dBqs_by_dcoeffs, arc_length)
        # res += np.einsum('i,ijm,ij->m', np.sum(diff**2, axis=1)/arc_length, dgammadash_by_dcoeff, gammadash)

        tmp = np.sum(dBbs_by_dX[:, :, :, None]*dgamma_by_dcoeff[:, :, None, :], axis=1)
        res = 2*np.einsum('ij,ijm,i->m', diff, tmp-dBqs_by_dcoeffs, arc_length)
        res += np.einsum('i,ijm,ij->m', np.sum(diff**2, axis=1)/arc_length, dgammadash_by_dcoeff, gammadash)

        res *= 1/arc_length.shape[0]
        return res

    def dJ_L2_by_detabar(self):
        Bbs             = self.biotsavart.B(compute_derivatives=2)
        arc_length      = self.quasi_symmetric_field.magnetic_axis.incremental_arclength()
        Bqs             = self.quasi_symmetric_field.B
        dBqs_by_detabar = self.quasi_symmetric_field.dB_by_detabar
        res = np.zeros((1, ))
        res[0] -= np.sum(2*(Bbs-Bqs)*dBqs_by_detabar[:, 0, :] * arc_length[:, None])
        res *= 1/arc_length.shape[0]
        return res

    def J_H1(self):
        dBbs_by_dX = self.biotsavart.dB_by_dX(compute_derivatives=(1 if self.value_only else 2))
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength()
        dBqs_by_dX = self.quasi_symmetric_field.dB_by_dX
        return np.sum(arc_length[:, None, None] * (dBbs_by_dX-dBqs_by_dX)**2)/len(arc_length)

    @writable_cached_property
    def dJ_H1_by_dcoilcoefficients(self):
        self.compute_dcoilcoefficients(reverse_mode=True)
        return self.dJ_H1_by_dcoilcoefficients

    def dJ_H1_by_dcoilcurrents(self):
        dBbs_by_dX               = self.biotsavart.dB_by_dX(compute_derivatives=2)
        d2Bbs_by_dXdcoilcurrents = self.biotsavart.d2B_by_dXdcoilcurrents()
        dBqs_by_dX               = self.quasi_symmetric_field.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength()
        res = []
        fak = 2/len(arc_length)
        tmp = fak*(dBbs_by_dX-dBqs_by_dX) * arc_length[:, None, None]
        return np.einsum('ijk,mijk->m', tmp, np.asarray(d2Bbs_by_dXdcoilcurrents))

    def dJ_H1_by_dmagneticaxiscoefficients(self):
        gamma                = self.quasi_symmetric_field.magnetic_axis.gamma()
        gammadash            = self.quasi_symmetric_field.magnetic_axis.gammadash()
        dgamma_by_dcoeff     = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dcoeff()
        dgammadash_by_dcoeff = self.quasi_symmetric_field.magnetic_axis.dgammadash_by_dcoeff()
        d2Bqs_by_dcoeffsdX   = self.quasi_symmetric_field.d2B_by_dcoeffsdX
        arc_length           = self.quasi_symmetric_field.magnetic_axis.incremental_arclength()

        dBbs_by_dX    = self.biotsavart.dB_by_dX(compute_derivatives=2)
        d2Bbs_by_dXdX = self.biotsavart.d2B_by_dXdX()
        dBqs_by_dX    = self.quasi_symmetric_field.dB_by_dX

        num_coeff = dgamma_by_dcoeff.shape[1]
        diff = dBbs_by_dX-dBqs_by_dX

        # res = 2 * np.einsum('ijk,ijlk,iml,i->m',(dBbs_by_dX-dBqs_by_dX), d2Bbs_by_dXdX, dgamma_by_dcoeff, arc_length)
        # res -= 2*np.einsum('ijk,imjk,i->m', (dBbs_by_dX-dBqs_by_dX), d2Bqs_by_dcoeffsdX, arc_length)
        # res += np.einsum('i,i,iml,il->m', (1/arc_length), np.sum(np.sum((dBbs_by_dX-dBqs_by_dX)**2, axis=1), axis=1), dgammadash_by_dcoeff, gammadash)

        tmp = np.sum(d2Bbs_by_dXdX[:,:,:,:,None]*dgamma_by_dcoeff[:, None, :, None, :], axis=2)
        res = 2 * np.einsum('ijk,ijkm,i->m',diff, tmp-d2Bqs_by_dcoeffsdX, arc_length)
        res += np.einsum('i,ilm,il->m', np.sum(diff**2, axis=(1, 2))/arc_length, dgammadash_by_dcoeff, gammadash)



        res *= 1/gamma.shape[0]
        return res

    def dJ_H1_by_detabar(self):
        dBbs_by_dX         = self.biotsavart.dB_by_dX(compute_derivatives=2)
        dBqs_by_dX         = self.quasi_symmetric_field.dB_by_dX
        d2Bqs_by_detabardX = self.quasi_symmetric_field.d2B_by_detabardX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength()

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
        quadrature_points = self.curve.gamma()
        arc_length = np.linalg.norm(self.curve.gammadash(), axis=1)
        B = self.biotsavart.set_points(quadrature_points).B()
        return np.sum(arc_length[:, None] * (B**2))/quadrature_points.shape[0]


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
        quadrature_points = self.curve.gamma()
        arc_length = np.linalg.norm(self.curve.gammadash(), axis=1)
        self.biotsavart.set_points(quadrature_points)
        dB_by_dX = self.biotsavart.dB_by_dX()
        return np.sum(arc_length * (np.sum(np.sum(dB_by_dX**2, axis=1), axis=1)))/quadrature_points.shape[0]



class CurveLength():

    r"""
    J = \int_{curve} 1 ds
    """

    def __init__(self, curve):
        self.curve = curve

    def J(self):
        arc_length = np.linalg.norm(self.curve.gammadash(), axis=1)
        return np.mean(arc_length)

    def dJ_by_dcoefficients(self):
        gammadash            = self.curve.gammadash()
        dgammadash_by_dcoeff = self.curve.dgammadash_by_dcoeff()
        num_coeff            = dgammadash_by_dcoeff.shape[2]
        res = np.zeros((num_coeff, ))
        arc_length = np.linalg.norm(gammadash, axis=1)
        for i in range(num_coeff):
            res[i] = np.mean((1/arc_length) * np.sum(dgammadash_by_dcoeff[:, :, i] * gammadash, axis=1))
        return res


class CurveCurvature():

    r"""
    J = \int_{curve} \kappa ds
    """

    def __init__(self, curve, desired_length=None, p=2, root=False):
        self.curve = curve
        if desired_length is None:
            self.desired_kappa = 0
        else:
            radius = desired_length/(2*pi)
            self.desired_kappa = 1/radius
        self.p = p
        self.root = root

    def J(self):
        p = self.p
        arc_length = np.linalg.norm(self.curve.gammadash(), axis=1)
        kappa = self.curve.kappa()
        if self.root:
            return np.mean(np.maximum(kappa-self.desired_kappa, 0)**p * arc_length)**(1./p)
        else:
            return np.mean(np.maximum(kappa-self.desired_kappa, 0)**p * arc_length)

    def dJ_by_dcoefficients(self):
        p = self.p
        kappa                = self.curve.kappa()
        dkappa_by_dcoeff     = self.curve.dkappa_by_dcoeff()
        gammadash            = self.curve.gammadash()
        dgammadash_by_dcoeff = self.curve.dgammadash_by_dcoeff()
        arc_length           = np.linalg.norm(gammadash, axis = 1)

        num_coeff = dgammadash_by_dcoeff.shape[2]
        res       = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i]  = np.mean((np.maximum(kappa-self.desired_kappa, 0)**p/arc_length) * np.sum(dgammadash_by_dcoeff[:, :, i] * gammadash, axis  = 1))
            res[i] += np.mean(p*(np.maximum(kappa-self.desired_kappa, 0))**(p-1) * dkappa_by_dcoeff[:,i] * arc_length)
            if self.root:
                res[i] *= (1./p) * np.mean(np.maximum(kappa-self.desired_kappa, 0)**p * arc_length)**(1./p-1)
        return res


class CurveTorsion():

    r"""
    J = \int_{curve} \tau^p ds
    """

    def __init__(self, curve, p=2, root=False):
        self.curve = curve
        self.p = p
        self.root = root

    def J(self):
        arc_length = np.linalg.norm(self.curve.gammadash(), axis=1)
        torsion    = self.curve.torsion()
        if self.root:
            return np.mean(np.abs(torsion)**self.p * arc_length)**(1./self.p)
        else:
            return np.mean(np.abs(torsion)**self.p * arc_length)

    def dJ_by_dcoefficients(self):
        torsion              = self.curve.torsion()
        dtorsion_by_dcoeff   = self.curve.dtorsion_by_dcoeff()
        gammadash            = self.curve.gammadash()
        dgammadash_by_dcoeff = self.curve.dgammadash_by_dcoeff()
        arc_length           = np.linalg.norm(gammadash, axis=1)

        num_coeff = dgammadash_by_dcoeff.shape[2]
        res       = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i]  = np.mean((np.abs(torsion)**self.p/arc_length) * np.sum(dgammadash_by_dcoeff[:, :, i] * gammadash, axis=1))
            res[i] += np.mean(self.p*np.abs(torsion)**(self.p-1) * np.sign(torsion) * dtorsion_by_dcoeff[:, i] * arc_length)
            if self.root:
                    res[i] *= (1./self.p) * np.mean(np.abs(torsion)**self.p * arc_length)**(1./self.p - 1.)
        return res


class SobolevTikhonov():

    def __init__(self, curve, weights=[1., 1., 0., 0.]):
        self.curve = curve
        if not len(weights) == 4:
            raise ValueError(
                "You should pass 4 weights: for the L^2, H^1, H^2 and H^3 norm.")
        self.weights = weights
        self.initial_curve = (
            curve.gamma().copy(), curve.gammadash().copy(),
            curve.gammadashdash().copy(), curve.gammadashdashdash().copy()
        )

    def J(self):
        res = 0
        curve = self.curve
        num_points = curve.gamma().shape[0]
        weights = self.weights
        if weights[0] > 0:
            res += weights[0] * \
                np.sum((curve.gamma()-self.initial_curve[0])**2)/num_points
        if weights[1] > 0:
            res += weights[1] * np.sum((curve.gammadash() -
                                        self.initial_curve[1])**2)/num_points
        if weights[2] > 0:
            res += weights[2] * np.sum((curve.gammadashdash() -
                                        self.initial_curve[2])**2)/num_points
        if weights[3] > 0:
            res += weights[3] * np.sum((curve.gammadashdashdash() -
                                        self.initial_curve[3])**2)/num_points
        return res

    def dJ_by_dcoefficients(self):
        curve = self.curve
        num_coeff = curve.dgamma_by_dcoeff().shape[2]
        num_points = curve.gamma().shape[0]
        res = np.zeros((num_coeff, ))
        weights = self.weights
        if weights[0] > 0:
            for i in range(num_coeff):
                res[i] += weights[0] * np.sum(
                    2*(curve.gamma()-self.initial_curve[0])*curve.dgamma_by_dcoeff()[:, :, i])/num_points
        if weights[1] > 0:
            for i in range(num_coeff):
                res[i] += weights[1] * np.sum(2*(curve.gammadash()-self.initial_curve[1])
                                              * curve.dgammadash_by_dcoeff()[:, :, i])/num_points
        if weights[2] > 0:
            for i in range(num_coeff):
                res[i] += weights[2] * np.sum(2*(curve.gammadashdash()-self.initial_curve[2])
                                              * curve.dgammadashdash_by_dcoeff()[:, :, i])/num_points
        if weights[3] > 0:
            for i in range(num_coeff):
                res[i] += weights[3] * np.sum(2*(curve.gammadashdashdash()-self.initial_curve[3])
                                              * curve.dgammadashdashdash_by_dcoeff()[:, :, i])/num_points
        return res


class UniformArclength():

    def __init__(self, curve, desired_length):
        self.curve = curve
        self.desired_arclength = desired_length

    def J(self):
        l = self.curve.incremental_arclength()
        num_points = l.shape[0]
        return np.mean((l-self.desired_arclength)**2)

    def dJ_by_dcoefficients(self):
        l = self.curve.incremental_arclength()
        dl = self.curve.dincremental_arclength_by_dcoeff()
        num_coeff = dl.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i] = np.mean(2 * (l-self.desired_arclength) * dl[:, i])
        return res


class MinimumDistance():

    def __init__(self, curves, minimum_distance):
        self.curves = curves
        self.minimum_distance = minimum_distance

    def min_dist(self):
        res = 1e10
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma()
            for j in range(i):
                gamma2 = self.curves[j].gamma()
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                res = min(res, np.min(dists))
        return res

    def J(self):
        from scipy.spatial.distance import cdist
        res = 0
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].incremental_arclength()[:, None]
            for j in range(i):
                gamma2 = self.curves[j].gamma()
                l2 = self.curves[j].incremental_arclength()[None, :]
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                alen = l1 * l2
                res += np.sum(alen * np.maximum(self.minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])
        return res

    def dJ_by_dcoefficients(self):
        res = []
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma()
            dgamma1 = self.curves[i].dgamma_by_dcoeff()
            numcoeff1 = dgamma1.shape[2]
            l1 = self.curves[i].incremental_arclength()[:, None]
            dl1 = self.curves[i].dincremental_arclength_by_dcoeff()[:, None, :]
            res.append(np.zeros((numcoeff1, )))
            for j in range(i):
                gamma2 = self.curves[j].gamma()
                dgamma2 = self.curves[j].dgamma_by_dcoeff()
                l2 = self.curves[j].incremental_arclength()[None, :]
                dl2 = self.curves[j].dincremental_arclength_by_dcoeff()[None, :, :]
                numcoeff2 = dgamma2.shape[2]
                diffs = gamma1[:, None, :] - gamma2[None, :, :]

                dists = np.sqrt(np.sum(diffs**2, axis=2))
                npmax = np.maximum(self.minimum_distance - dists, 0)
                if np.sum(npmax) < 1e-15:
                    continue

                l1l2npmax = l1*l2*npmax
                for ii in range(numcoeff1):
                    res[i][ii] += np.sum(dl1[:, :, ii] * l2 * npmax**2)/(gamma1.shape[0]*gamma2.shape[0])
                    res[i][ii] += np.sum(-2 * l1l2npmax * np.sum(dgamma1[:, None, :, ii] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])
                for jj in range(numcoeff2):
                    res[j][jj] += np.sum(l1 * dl2[:, :, jj] * npmax**2)/(gamma1.shape[0]*gamma2.shape[0])
                    res[j][jj] -= np.sum(-2 * l1l2npmax * np.sum(dgamma2[:, :, jj][None, :, :] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])
        return res

class CoilLpReduction():

    def __init__(self, objectives, p=2, root=False):
        self.objectives = objectives
        self.p = p
        self.root = root

    def J(self):
        p = self.p
        if self.root:
            return sum([J.J()**p for J in self.objectives])**(1./p)
        else:
            return sum([J.J()**p for J in self.objectives])

    def dJ_by_dcoefficients(self):
        p = self.p
        if self.root:
            return (1./p)*sum([J.J()**p for J in self.objectives])**(1./p-1) * np.concatenate([p*(J.J()**(p-1))*J.dJ_by_dcoefficients() for J in self.objectives], axis=0)
        else:
            return np.concatenate([p*(J.J()**(p-1))*J.dJ_by_dcoefficients() for J in self.objectives], axis=0)
