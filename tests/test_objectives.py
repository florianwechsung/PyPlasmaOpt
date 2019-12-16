import numpy as np
import pytest
from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve, \
    SquaredMagneticFieldNormOnCurve, SquaredMagneticFieldGradientNormOnCurve, get_matt_data, CoilCollection, \
    QuasiSymmetricField, BiotSavartQuasiSymmetricFieldDifference


@pytest.mark.parametrize("gradient", [True, False])
def test_magnetic_field_objective_by_dcoilcoeffs(gradient):
    nfp = 2
    (coils, ma) = get_matt_data(nfp=nfp)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    if gradient:
        J = SquaredMagneticFieldGradientNormOnCurve(ma, bs)
    else:
        J = SquaredMagneticFieldNormOnCurve(ma, bs)
    J0 = J.J()
    coil_dofs = stellerator.get_dofs()
    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = stellerator.reduce_coefficient_derivatives(J.dJ_by_dcoilcoefficients())
    assert len(dJ) == len(h)
    deriv = np.sum(dJ * h)
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        stellerator.set_dofs(coil_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

@pytest.mark.parametrize("gradient", [True, False])
def test_magnetic_field_objective_by_dcurvecoeffs(gradient):
    nfp = 2
    (coils, ma) = get_matt_data(nfp=nfp)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    if gradient:
        J = SquaredMagneticFieldGradientNormOnCurve(ma, bs)
    else:
        J = SquaredMagneticFieldNormOnCurve(ma, bs)
    J0 = J.J()
    curve_dofs = ma.get_dofs()
    h = 1e-1 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
    dJ = J.dJ_by_dcurvecoefficients()
    deriv = np.sum(dJ * h)
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        ma.set_dofs(curve_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

@pytest.mark.parametrize("objective", ["l2", "h1"])
def test_taylor_test_coil_coeffs(objective):
    num_coils = 6
    nfp = 2
    coils, ma = get_matt_data(Nt=4, nfp=nfp, ppp=20)
    currents = num_coils * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    qsf = QuasiSymmetricField(-2.25, ma)
    sigma, iota = qsf.solve_state()
    J = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
    J.update()
    coil_dofs = stellerator.get_dofs()
    h =1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    if objective == "l2":
        J0 = J.J_L2()
        dJ = stellerator.reduce_coefficient_derivatives(J.dJ_L2_by_dcoilcoefficients())
    else:
        J0 = J.J_H1()
        dJ = stellerator.reduce_coefficient_derivatives(J.dJ_H1_by_dcoilcoefficients())
    assert len(dJ) == len(h)
    deriv = np.sum(dJ * h)
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        stellerator.set_dofs(coil_dofs + eps * h)
        J.update()
        Jh = J.J_L2() if objective == "l2" else J.J_H1()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

@pytest.mark.parametrize("objective", ["l2", "h1"])
def test_taylor_test_coil_currents(objective):
    num_coils = 6
    nfp = 2
    coils, ma = get_matt_data(Nt=4, nfp=nfp, ppp=20)
    currents = num_coils * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    qsf = QuasiSymmetricField(-2.25, ma)
    sigma, iota = qsf.solve_state()
    J = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
    J.update()
    x0 = stellerator.get_currents()
    h = 1e4 * np.random.rand(len(currents))
    if objective == "l2":
        J0 = J.J_L2()
        dJ = stellerator.reduce_current_derivatives(J.dJ_L2_by_dcoilcurrents())
    else:
        J0 = J.J_H1()
        dJ = stellerator.reduce_current_derivatives(J.dJ_H1_by_dcoilcurrents())
    assert len(dJ) == len(h)
    deriv = np.sum(dJ * h)
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        stellerator.set_currents(x0 + eps * h)
        J.update()
        Jh = J.J_L2() if objective == "l2" else J.J_H1()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new
