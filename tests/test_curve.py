import numpy as np
import pytest

def taylor_test(f, df, x, epsilons=None, direction=None):
    f0 = f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = direction@df(x)
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(10, 15)))
    print("################################################################################")
    err_old = 1e9
    for eps in epsilons:
        feps = f(x + eps * direction)
        dfest = (feps-f0)/eps
        err = np.linalg.norm(dfest - dfx)
        assert err < 1e-11 or err < 0.6 * err_old
        err_old = err
        print(err)
    print("################################################################################")

def get_coil():
    from pyplasmaopt import CartesianFourierCurve

    cfc = CartesianFourierCurve(3)
    cfc.coefficients[1][0] = 1.
    cfc.coefficients[1][1] = 0.5
    cfc.coefficients[2][2] = 0.5
    return cfc

def test_coil_first_derivative():
    cfc = get_coil()
    x = np.asarray([1])
    taylor_test(lambda p: cfc.gamma(p), lambda p: cfc.dgamma_by_dphi(p), x)

def test_coil_dof_numbering():
    cfc = get_coil()
    coeffs = cfc.get_dofs()
    cfc.set_dofs(coeffs)
    assert(np.allclose(coeffs, cfc.get_dofs()))

def test_coil_coefficient_derivative():
    cfc = get_coil()
    coeffs = cfc.get_dofs()
    x = np.asarray([1])
    def f(dofs):
        cfc.set_dofs(dofs)
        return cfc.gamma(x)
    def df(dofs):
        cfc.set_dofs(dofs)
        return cfc.dgamma_by_dcoeff(x)
    taylor_test(f, df, coeffs)

def test_coil_curvature_derivative():
    # This implicitly also tests the higher order derivatives of gamma as these
    # are needed to compute the derivative of the curvature.
    cfc = get_coil()
    coeffs = cfc.get_dofs()
    x = np.asarray([1])
    def f(dofs):
        cfc.set_dofs(dofs)
        return cfc.kappa(x)
    def df(dofs):
        cfc.set_dofs(dofs)
        return cfc.dkappa_by_dcoeff(x)
    taylor_test(f, df, coeffs)

def get_magnetic_axis():
    from pyplasmaopt import StelleratorSymmetricCylindricalFourierCurve

    ma = StelleratorSymmetricCylindricalFourierCurve(3, 2)
    ma.coefficients[0][0] = 1.
    ma.coefficients[0][1] = 0.1
    ma.coefficients[1][0] = 0.1
    return ma

def test_magnetic_axis_first_derivative():
    ma = get_magnetic_axis()
    x = np.asarray([1])
    taylor_test(lambda p: ma.gamma(p), lambda p: ma.dgamma_by_dphi(p), x)

def test_magnetic_axis_dof_numbering():
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    ma.set_dofs(coeffs)
    assert(np.allclose(coeffs, ma.get_dofs()))

def test_magnetic_axis_coefficient_derivative():
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    x = np.asarray([1])
    def f(dofs):
        ma.set_dofs(dofs)
        return ma.gamma(x)
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.dgamma_by_dcoeff(x)
    taylor_test(f, df, coeffs)

def test_magnetic_axis_curvature_derivative():
    # This implicitly also tests the higher order derivatives of gamma as these
    # are needed to compute the derivative of the curvature.
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    x = np.asarray([1])
    def f(dofs):
        ma.set_dofs(dofs)
        return ma.kappa(x)
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.dkappa_by_dcoeff(x)
    taylor_test(f, df, coeffs)

def test_magnetic_axis_incremental_arclength_derivative():
    # This implicitly also tests the higher order derivatives of gamma as these
    # are needed to compute the derivative of the curvature.
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    x = np.asarray([1])
    def f(dofs):
        ma.set_dofs(dofs)
        return ma.incremental_arclength(x)
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.dincremental_arclength_by_dcoeff(x)
    taylor_test(f, df, coeffs)

def test_magnetic_axis_frenet_frame():
    ma = get_magnetic_axis()
    x = np.asarray([0.1, 1.])
    (t, n, b) = ma.frenet_frame(x)
    assert np.allclose(np.sum(n*t, axis=1), 0)
    assert np.allclose(np.sum(n*b, axis=1), 0)
    assert np.allclose(np.sum(t*b, axis=1), 0)
    assert np.allclose(np.sum(t*t, axis=1), 1)
    assert np.allclose(np.sum(n*n, axis=1), 1)
    assert np.allclose(np.sum(b*b, axis=1), 1)

if __name__ == "__main__":
    cfc = get_coil()
    ax = cfc.plot(plot_derivative=True, show=False)
    ma = get_magnetic_axis()
    ax = ma.plot(ax=ax, plot_derivative=False, show=False)
    points = np.linspace(0, 1, 100)
    x = ma.gamma(points)
    (t, n, b) = ma.frenet_frame(points)
    ax.quiver(x[:, 0], x[:, 1], x[:, 2], t[:, 0], t[:, 1], t[:, 2], arrow_length_ratio=0.3, color="b", length=0.2)
    ax.quiver(x[:, 0], x[:, 1], x[:, 2], n[:, 0], n[:, 1], n[:, 2], arrow_length_ratio=0.3, color="g", length=0.2)
    ax.quiver(x[:, 0], x[:, 1], x[:, 2], b[:, 0], b[:, 1], b[:, 2], arrow_length_ratio=0.3, color="y", length=0.2)
    import matplotlib.pyplot as plt
    plt.show()


