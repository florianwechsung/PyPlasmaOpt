import pytest
import numpy as np
from pyplasmaopt import CartesianFourierCurve, CurveLength, CurveCurvature

def get_coil():
    coil = CartesianFourierCurve(3, np.linspace(0, 1, 20, endpoint=False))
    coil.coefficients[1][0] = 1.
    coil.coefficients[1][1] = 0.5
    coil.coefficients[2][2] = 0.5
    return coil

def test_curve_length_taylor_test():
    coil = get_coil()
    J = CurveLength(coil)
    J0 = J.J()
    coil_dofs = coil.get_dofs()
    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = J.dJ_dcoefficients()
    deriv = np.sum(dJ * h)
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new


def test_curve_curvature_taylor_test():
    coil = get_coil()
    J = CurveCurvature(coil)
    J0 = J.J()
    coil_dofs = coil.get_dofs()
    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = J.dJ_dcoefficients()
    deriv = np.sum(dJ * h)
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

