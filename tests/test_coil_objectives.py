import pytest
import numpy as np
from pyplasmaopt import CartesianFourierCurve, CurveLength, CurveCurvature, CurveTorsion, MinimumDistance, get_24_coil_data, CoilCollection, SobolevTikhonov, UniformArclength

def get_coil(rand_scale=0.01):
    coil = CartesianFourierCurve(3, np.linspace(0, 1, 20, endpoint=False))
    coil.coefficients[1][0] = 1.
    coil.coefficients[1][1] = 0.5
    coil.coefficients[2][2] = 0.5
    dofs = coil.get_dofs()
    coil.set_dofs(dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape))
    return coil

def test_curve_length_taylor_test():
    coil = get_coil()
    J = CurveLength(coil)
    J0 = J.J()
    coil_dofs = coil.get_dofs()
    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = J.dJ_by_dcoefficients()
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
    from math import pi
    J = CurveCurvature(coil, desired_length=2*pi/np.mean(coil.kappa))
    J0 = J.J()
    coil_dofs = coil.get_dofs()
    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = J.dJ_by_dcoefficients()
    deriv = np.sum(dJ * h)
    assert deriv > 1e-10
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

def test_curve_torsion_taylor_test():
    coil = get_coil()
    J = CurveTorsion(coil)
    J0 = J.J()
    coil_dofs = coil.get_dofs()
    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = J.dJ_by_dcoefficients()
    deriv = np.sum(dJ * h)
    err = 1e8
    for i in range(5, 10):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

def test_sobolev_tikhonov_taylor_test():
    coil = get_coil()
    J = SobolevTikhonov(coil, weights=[1., 2., 3., 4.])
    coil_dofs = coil.get_dofs()
    perturb = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    coil.set_dofs(coil_dofs + perturb)
    coil_dofs = coil.get_dofs()
    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    J0 = J.J()
    dJ = J.dJ_by_dcoefficients()
    deriv = np.sum(dJ * h)
    err = 1e8
    for i in range(5, 10):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

def test_uniformarclength_taylor_test():
    coil = get_coil()
    J = UniformArclength(coil, np.mean(coil.incremental_arclength))
    coil_dofs = coil.get_dofs()
    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    J0 = J.J()
    dJ = J.dJ_by_dcoefficients()
    deriv = np.sum(dJ * h)
    err = 1e8
    for i in range(5, 10):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new


def test_minimum_distance_taylor_test():
    nfp = 2
    (coils, currents, ma, eta_bar) = get_24_coil_data(nfp=nfp)
    currents = len(coils) * [1e4]
    stellarator = CoilCollection(coils, currents, nfp, True)
    coils = stellarator.coils
    np.random.seed(2)
    J = MinimumDistance(coils, 0.1)
    coil_dofs = stellarator.get_dofs()
    h = np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = stellarator.reduce_coefficient_derivatives(J.dJ_by_dcoefficients())
    deriv = np.sum(dJ * h)
    err = 1e8
    for i in range(1, 5):
        eps = 0.1**i
        stellarator.set_dofs(coil_dofs + eps * h)
        Jp = J.J()
        stellarator.set_dofs(coil_dofs - eps * h)
        Jm = J.J()
        deriv_est = 0.5 * (Jp-Jm)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55**2 * err
        err = err_new
    print("deriv_est %s" % (deriv_est))
