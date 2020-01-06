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

def get_coil(x=np.asarray([0.5])):
    from pyplasmaopt import CartesianFourierCurve

    cfc = CartesianFourierCurve(3, x)
    cfc.coefficients[1][0] = 1.
    cfc.coefficients[1][1] = 0.5
    cfc.coefficients[2][2] = 0.5
    cfc.update()
    return cfc

def test_coil_first_derivative():
    h = 0.1
    epss = [0.5**i for i in range(5, 10)] 
    x = np.asarray([1] + [1 + eps for eps in epss])
    cfc = get_coil(x)
    f0 = cfc.gamma[0]
    deriv = cfc.dgamma_by_dphi[0, 0]
    err_old = 1e6
    for i in range(len(epss)):
        fh = cfc.gamma[i+1]
        deriv_est = (fh-f0)/epss[i]
        err = np.linalg.norm(deriv_est-deriv)
        assert err < 0.55 * err_old
        err_old = err

def test_coil_second_derivative():
    h = 0.1
    epss = [0.5**i for i in range(5, 10)] 
    x = np.asarray([1] + [1 + eps for eps in epss])
    cfc = get_coil(x)
    f0 = cfc.dgamma_by_dphi[0, 0, :]
    deriv = cfc.d2gamma_by_dphidphi[0, 0, 0, :]
    err_old = 1e6
    for i in range(len(epss)):
        fh = cfc.dgamma_by_dphi[i+1, 0, :]
        deriv_est = (fh-f0)/epss[i]
        err = np.linalg.norm(deriv_est-deriv)
        assert err < 0.55 * err_old
        err_old = err

def test_coil_third_derivative():
    h = 0.1
    epss = [0.5**i for i in range(5, 10)] 
    x = np.asarray([1] + [1 + eps for eps in epss])
    cfc = get_coil(x)
    f0 = cfc.d2gamma_by_dphidphi[0, 0, 0, :]
    deriv = cfc.d3gamma_by_dphidphidphi[0, 0, 0, 0, :]
    err_old = 1e6
    for i in range(len(epss)):
        fh = cfc.d2gamma_by_dphidphi[i+1, 0, 0, :]
        deriv_est = (fh-f0)/epss[i]
        err = np.linalg.norm(deriv_est-deriv)
        assert err < 0.55 * err_old
        err_old = err

def test_coil_dof_numbering():
    cfc = get_coil()
    coeffs = cfc.get_dofs()
    cfc.set_dofs(coeffs)
    assert(np.allclose(coeffs, cfc.get_dofs()))

def test_coil_coefficient_derivative():
    cfc = get_coil()
    coeffs = cfc.get_dofs()
    def f(dofs):
        cfc.set_dofs(dofs)
        return cfc.gamma.copy()
    def df(dofs):
        cfc.set_dofs(dofs)
        return cfc.dgamma_by_dcoeff.copy()
    taylor_test(f, df, coeffs)

    def f(dofs):
        cfc.set_dofs(dofs)
        return cfc.dgamma_by_dphi[:, 0, :].copy()
    def df(dofs):
        cfc.set_dofs(dofs)
        return cfc.d2gamma_by_dphidcoeff[:, 0, :, :].copy()
    taylor_test(f, df, coeffs)

def test_coil_curvature_derivative():
    # This implicitly also tests the higher order derivatives of gamma as these
    # are needed to compute the derivative of the curvature.
    cfc = get_coil()
    coeffs = cfc.get_dofs()
    def f(dofs):
        cfc.set_dofs(dofs)
        return cfc.kappa.copy()
    def df(dofs):
        cfc.set_dofs(dofs)
        return cfc.dkappa_by_dcoeff.copy()
    taylor_test(f, df, coeffs)

def get_magnetic_axis(x=np.asarray([0.12345])):
    from pyplasmaopt import StelleratorSymmetricCylindricalFourierCurve

    ma = StelleratorSymmetricCylindricalFourierCurve(3, 2, x)
    ma.coefficients[0][0] = 1.
    ma.coefficients[0][1] = 0.1
    ma.coefficients[1][0] = 0.1
    ma.update()
    return ma

def test_magnetic_axis_first_derivative():
    h = 0.1
    epss = [0.5**i for i in range(5, 10)] 
    x = np.asarray([1] + [1 + eps for eps in epss])
    ma = get_magnetic_axis(x)
    f0 = ma.gamma[0]
    deriv = ma.dgamma_by_dphi[0, 0]
    err_old = 1e6
    for i in range(len(epss)):
        fh = ma.gamma[i+1]
        deriv_est = (fh-f0)/epss[i]
        err = np.linalg.norm(deriv_est-deriv)
        assert err < 0.55 * err_old
        err_old = err

def test_magnetic_axis_second_derivative():
    h = 0.1
    epss = [0.5**i for i in range(5, 10)] 
    x = np.asarray([0.1234] + [0.1234 + eps for eps in epss])
    ma = get_magnetic_axis(x)
    f0 = ma.dgamma_by_dphi[0, 0, :]
    deriv = ma.d2gamma_by_dphidphi[0, 0, 0, :]
    err_old = 1e6
    for i in range(len(epss)):
        fh = ma.dgamma_by_dphi[i+1, 0, :]
        deriv_est = (fh-f0)/epss[i]
        err = np.linalg.norm(deriv_est-deriv)
        assert err < 0.55 * err_old
        err_old = err

def test_magnetic_axis_third_derivative():
    h = 0.1
    epss = [0.5**i for i in range(5, 10)] 
    x = np.asarray([0.1234] + [0.1234 + eps for eps in epss])
    ma = get_magnetic_axis(x)
    f0 = ma.d2gamma_by_dphidphi[0, 0, 0, :]
    deriv = ma.d3gamma_by_dphidphidphi[0, 0, 0, 0, :]
    err_old = 1e6
    for i in range(len(epss)):
        fh = ma.d2gamma_by_dphidphi[i+1, 0, 0, :]
        deriv_est = (fh-f0)/epss[i]
        err = np.linalg.norm(deriv_est-deriv)
        assert err < 0.55 * err_old
        err_old = err

def test_magnetic_axis_kappa_first_derivative():
    h = 0.1
    epss = [0.5**i for i in range(5, 10)] 
    x = np.asarray([0.1234] + [0.1234 + eps for eps in epss])
    ma = get_magnetic_axis(x)
    f0 = ma.kappa[0]
    deriv = ma.dkappa_by_dphi[0, 0]
    err_old = 1e6
    print(deriv)
    for i in range(len(epss)):
        fh = ma.kappa[i+1]
        deriv_est = (fh-f0)/epss[i]
        print(deriv_est)
        err = np.linalg.norm(deriv_est-deriv)
        assert err < 0.55 * err_old
        err_old = err

def test_magnetic_axis_kappa_derivative():
    cfc = get_coil()
    coeffs = cfc.get_dofs()
    def f(dofs):
        cfc.set_dofs(dofs)
        return cfc.kappa.copy()
    def df(dofs):
        cfc.set_dofs(dofs)
        return cfc.dkappa_by_dcoeff.copy()
    taylor_test(f, df, coeffs)

def test_magnetic_axis_torsion_derivative():
    cfc = get_coil()
    coeffs = cfc.get_dofs()
    def f(dofs):
        cfc.set_dofs(dofs)
        return cfc.torsion.copy()
    def df(dofs):
        cfc.set_dofs(dofs)
        return cfc.dtorsion_by_dcoeff.copy()
    taylor_test(f, df, coeffs)

def test_magnetic_axis_dof_numbering():
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    ma.set_dofs(coeffs)
    assert(np.allclose(coeffs, ma.get_dofs()))

def test_magnetic_axis_coefficient_derivative():
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    def f(dofs):
        ma.set_dofs(dofs)
        return ma.gamma.copy()
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.dgamma_by_dcoeff.copy()
    taylor_test(f, df, coeffs)

def test_magnetic_axis_curvature_derivative():
    # This implicitly also tests the higher order derivatives of gamma as these
    # are needed to compute the derivative of the curvature.
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    def f(dofs):
        ma.set_dofs(dofs)
        return ma.kappa.copy()
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.dkappa_by_dcoeff.copy()
    taylor_test(f, df, coeffs)

def test_magnetic_axis_incremental_arclength_derivative():
    # This implicitly also tests the higher order derivatives of gamma as these
    # are needed to compute the derivative of the curvature.
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    def f(dofs):
        ma.set_dofs(dofs)
        return ma.incremental_arclength.copy()
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.dincremental_arclength_by_dcoeff.copy()
    taylor_test(f, df, coeffs)

def test_magnetic_axis_frenet_frame_derivative():
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    def f(dofs):
        ma.set_dofs(dofs)
        return ma.frenet_frame[0].copy()
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.dfrenet_frame_by_dcoeff[0].copy()
    taylor_test(f, df, coeffs)

    def f(dofs):
        ma.set_dofs(dofs)
        return ma.frenet_frame[1].copy()
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.dfrenet_frame_by_dcoeff[1].copy()
    taylor_test(f, df, coeffs)

    def f(dofs):
        ma.set_dofs(dofs)
        return ma.frenet_frame[2].copy()
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.dfrenet_frame_by_dcoeff[2].copy()
    taylor_test(f, df, coeffs)

def test_magnetic_axis_dkappa_by_dphi_derivative():
    ma = get_magnetic_axis()
    coeffs = ma.get_dofs()
    def f(dofs):
        ma.set_dofs(dofs)
        return ma.dkappa_by_dphi[:,0,:].copy()
    def df(dofs):
        ma.set_dofs(dofs)
        return ma.d2kappa_by_dphidcoeff[:, 0, :, :].copy()
    taylor_test(f, df, coeffs)


def test_magnetic_axis_frenet_frame():
    ma = get_magnetic_axis()
    (t, n, b) = ma.frenet_frame
    assert np.allclose(np.sum(n*t, axis=1), 0)
    assert np.allclose(np.sum(n*b, axis=1), 0)
    assert np.allclose(np.sum(t*b, axis=1), 0)
    assert np.allclose(np.sum(t*t, axis=1), 1)
    assert np.allclose(np.sum(n*n, axis=1), 1)
    assert np.allclose(np.sum(b*b, axis=1), 1)

if __name__ == "__main__":
    points = np.linspace(0, 1, 100)
    cfc = get_coil(points)
    ax = cfc.plot(plot_derivative=True, show=False)
    ma = get_magnetic_axis(points)
    ax = ma.plot(ax=ax, plot_derivative=False, show=False)
    x = ma.gamma
    (t, n, b) = ma.frenet_frame
    ax.quiver(x[:, 0], x[:, 1], x[:, 2], t[:, 0], t[:, 1], t[:, 2], arrow_length_ratio=0.3, color="b", length=0.2)
    ax.quiver(x[:, 0], x[:, 1], x[:, 2], n[:, 0], n[:, 1], n[:, 2], arrow_length_ratio=0.3, color="g", length=0.2)
    ax.quiver(x[:, 0], x[:, 1], x[:, 2], b[:, 0], b[:, 1], b[:, 2], arrow_length_ratio=0.3, color="y", length=0.2)
    import matplotlib.pyplot as plt
    plt.show()
