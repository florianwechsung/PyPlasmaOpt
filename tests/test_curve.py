import numpy as np
import pytest

def taylor_test(f, df, x, epsilons=None, direction=None):
    f0 = f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))
    dfx = direction@df(x)
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(5, 10)))
    print("################################################################################")
    err_old = 1e9
    for eps in epsilons:
        feps = f(x + eps * direction)
        dfest = (feps-f0)/eps
        err = np.linalg.norm(dfest - dfx)
        assert err < 1e-12 or err < 0.6 * err_old
        err_old = err
        print(err)
    print("################################################################################")



def get_curve():
    from pyplasmaopt import CartesianFourierCurve

    cfc = CartesianFourierCurve(3)
    cfc.coefficients[1][1] = 1.
    cfc.coefficients[2][2] = 1.
    return cfc

def test_check_first_derivative():
    cfc = get_curve()
    x = np.asarray([1])
    taylor_test(lambda p: cfc.gamma(p), lambda p: cfc.dgamma_by_dphi(p), x)

def test_dof_numbering():
    cfc = get_curve()
    coeffs = cfc.get_dofs()
    cfc.set_dofs(coeffs)
    assert(np.allclose(coeffs, cfc.get_dofs()))

def test_coefficient_derivative():
    cfc = get_curve()
    coeffs = cfc.get_dofs()
    x = np.asarray([1])
    def f(dofs):
        cfc.set_dofs(dofs)
        return cfc.gamma(x)
    def df(dofs):
        cfc.set_dofs(dofs)
        return cfc.dgamma_by_dcoeff(x)
    taylor_test(f, df, coeffs)

def test_curvature_derivative():
    # This implicitly also tests the higher order derivatives of gamma as these
    # are needed to compute the derivative of the curvature.
    cfc = get_curve()
    coeffs = cfc.get_dofs()
    x = np.asarray([1])
    def f(dofs):
        cfc.set_dofs(dofs)
        return cfc.kappa(x)
    def df(dofs):
        cfc.set_dofs(dofs)
        return cfc.dkappa_by_dcoeff(x)
    taylor_test(f, df, coeffs)


if __name__ == "__main__":
    cfc = get_curve()
    cfc.plot(plot_derivative=True)

