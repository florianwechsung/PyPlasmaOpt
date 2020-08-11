import numpy as np
import pytest
from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve, \
    SquaredMagneticFieldNormOnCurve, SquaredMagneticFieldGradientNormOnCurve, get_24_coil_data, CoilCollection, \
    QuasiSymmetricField, BiotSavartQuasiSymmetricFieldDifference


@pytest.mark.parametrize("gradient", [True, False])
def test_magnetic_field_objective_by_dcoilcoeffs(gradient):
    nfp = 2
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
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
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
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
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    bs.set_points(ma.gamma)
    qsf = QuasiSymmetricField(-2.25, ma)
    J = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
    coil_dofs = stellerator.get_dofs()
    np.random.seed(1)
    h = np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    if objective == "l2":
        J0 = J.J_L2()
        dJ = stellerator.reduce_coefficient_derivatives(J.dJ_L2_by_dcoilcoefficients())
    else:
        J0 = J.J_H1()
        dJ = stellerator.reduce_coefficient_derivatives(J.dJ_H1_by_dcoilcoefficients())
    assert len(dJ) == len(h)
    deriv = np.sum(dJ * h)
    err = 1e6
    eps = 0.02
    while err > 1e-9:
        eps *= 0.5

        stellerator.set_dofs(coil_dofs + eps * h)
        bs.clear_cached_properties()
        Jhp = J.J_L2() if objective == "l2" else J.J_H1()

        stellerator.set_dofs(coil_dofs - eps * h)
        bs.clear_cached_properties()
        Jhm = J.J_L2() if objective == "l2" else J.J_H1()

        deriv_est = (Jhp-Jhm)/(2*eps)
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new
    assert eps < 1e-3

@pytest.mark.parametrize("objective", ["l2", "h1"])
def test_taylor_test_coil_currents(objective):
    num_coils = 6
    nfp = 2
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    bs.set_points(ma.gamma)
    qsf = QuasiSymmetricField(-2.25, ma)
    J = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
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
    eps = 0.1
    while err > 1e-7:
        eps *= 0.5
        stellerator.set_currents(x0 + eps * h)
        bs.clear_cached_properties()
        Jh = J.J_L2() if objective == "l2" else J.J_H1()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new
    assert eps < 1e-2

@pytest.mark.parametrize("objective", ["l2", "h1"])
def test_taylor_test_ma_coeffs(objective):
    num_coils = 6
    nfp = 2
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    bs.set_points(ma.gamma)
    qsf = QuasiSymmetricField(-2.25, ma)
    J = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
    ma_dofs = ma.get_dofs()
    np.random.seed(1)
    h = np.random.rand(len(ma_dofs)).reshape(ma_dofs.shape)
    if objective == "l2":
        J0 = J.J_L2()
        dJ = J.dJ_L2_by_dmagneticaxiscoefficients()
    else:
        J0 = J.J_H1()
        dJ = J.dJ_H1_by_dmagneticaxiscoefficients()
    assert len(dJ) == len(h)
    deriv = np.sum(dJ * h)
    err = 1e6
    eps = 0.04
    while err > 1e-11:
        eps *= 0.5
        deriv_est = 0
        shifts = [-2, -1, +1, +2]
        weights = [+1/12, -2/3, +2/3, -1/12]
        for i in range(4):
            ma.set_dofs(ma_dofs + shifts[i]*eps*h)
            bs.set_points(ma.gamma)
            qsf.clear_cached_properties()
            deriv_est += weights[i] * (J.J_L2() if objective == "l2" else J.J_H1())
        deriv_est *= 1/eps
        err_new = np.linalg.norm(deriv_est-deriv)/np.linalg.norm(deriv)
        print("err_new %s" % (err_new))
        assert err_new < (0.55)**4 * err
        err = err_new
    assert eps < 1e-3

def test_taylor_test_iota_by_coeffs():
    nfp = 2
    (_, _, ma, eta_bar) = get_24_coil_data(nfp=nfp, ppp=20)
    qsf = QuasiSymmetricField(eta_bar, ma)
    ma_dofs = ma.get_dofs()
    np.random.seed(1)
    h = 1e-1 * np.random.rand(len(ma_dofs)).reshape(ma_dofs.shape)
    qsf.solve_state()
    dJ = qsf.diota_by_dcoeffs
    assert len(dJ) == len(h)
    deriv = np.sum(dJ[:,0] * h)
    err = 1e6
    eps = 0.01
    while err > 1e-8:
        eps *= 0.5
        ma.set_dofs(ma_dofs + eps * h)
        qsf.clear_cached_properties()
        Jh = qsf.iota
        ma.set_dofs(ma_dofs - eps * h)
        qsf.clear_cached_properties()
        Jhm = qsf.iota
        deriv_est = (Jh-Jhm)/(2*eps)
        err_new = np.linalg.norm(deriv_est-deriv)/np.linalg.norm(deriv)
        assert err_new < 0.26 * err
        err = err_new
        print("err_new %s" % (err_new))
    assert eps < 1e-2

def test_taylor_test_sigma_by_coeffs():
    nfp = 2
    (_, _, ma, eta_bar) = get_24_coil_data(nfp=nfp, ppp=20)
    qsf = QuasiSymmetricField(eta_bar, ma)
    ma_dofs = ma.get_dofs()
    np.random.seed(1)
    h = 1e-1 * np.random.rand(len(ma_dofs)).reshape(ma_dofs.shape)
    qsf.solve_state()
    dJ = qsf.dsigma_by_dcoeffs
    deriv = dJ @ h
    err = 1e6
    eps = 0.01
    while err > 1e-8:
        eps *= 0.5

        ma.set_dofs(ma_dofs + eps * h)
        qsf.clear_cached_properties()
        Jhp = qsf.sigma

        ma.set_dofs(ma_dofs - eps * h)
        qsf.clear_cached_properties()
        Jhm = qsf.sigma

        deriv_est = (Jhp-Jhm)/(2*eps)
        err_new = np.linalg.norm(deriv_est-deriv)/np.linalg.norm(deriv)
        assert err_new < 0.26 * err
        err = err_new
        print("err_new %s" % (err_new))
    assert eps < 1e-2

if __name__ == "__main__":
    test_taylor_test_ma_coeffs("l2")
    test_taylor_test_ma_coeffs("h1")
    # test_taylor_test_sigma_by_coeffs()
