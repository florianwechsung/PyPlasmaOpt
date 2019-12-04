import numpy as np
import pytest
from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve, \
    SquaredMagneticFieldNormOnCurve, SquaredMagneticFieldGradientNormOnCurve, get_matt_data, CoilCollection

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
    dJ = stellerator.reduce_derivatives(J.dJ_by_dcoilcoefficients())
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

if __name__ == "__main__":
    test_magnetic_field_objective_by_dcoilcoeffs()
    test_magnetic_field_objective_by_dcurvecoeffs()
