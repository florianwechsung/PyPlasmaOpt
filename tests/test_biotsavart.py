import numpy as np
import pytest
from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve

def get_coil(num_quadrature_points=200):
    coil = CartesianFourierCurve(3, np.linspace(0, 1, num_quadrature_points, endpoint=False))
    coil.coefficients[1][0] = 1.
    coil.coefficients[1][1] = 0.5
    coil.coefficients[2][2] = 0.5
    coil.update()
    return coil

@pytest.mark.parametrize("use_cpp", [True, False])
def test_biotsavart_exponential_convergence(use_cpp):
    coil = get_coil()
    points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    btrue   = BiotSavart([get_coil(1000)], [1e4]).compute(points, use_cpp=use_cpp).B
    bcoarse = BiotSavart([get_coil(10)]  , [1e4]).compute(points, use_cpp=use_cpp).B
    bfine   = BiotSavart([get_coil(20)]  , [1e4]).compute(points, use_cpp=use_cpp).B
    assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)

    dbtrue   = BiotSavart([get_coil(1000)], [1e4]).compute(points, use_cpp=use_cpp).dB_by_dX
    dbcoarse = BiotSavart([get_coil(10)]  , [1e4]).compute(points, use_cpp=use_cpp).dB_by_dX
    dbfine   = BiotSavart([get_coil(20)]  , [1e4]).compute(points, use_cpp=use_cpp).dB_by_dX
    assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)

@pytest.mark.parametrize("use_cpp", [True, False])
@pytest.mark.parametrize("idx", [0, 16])
def test_biotsavart_dBdX_taylortest(use_cpp, idx):
    coil = get_coil()
    bs = BiotSavart([coil], [1e4])
    points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    points += 0.001 * (np.random.rand(*points.shape)-0.5)
    bs.compute(points, use_cpp=use_cpp)
    B0 = bs.B[idx]
    dB = bs.dB_by_dX[idx]
    for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
        deriv = dB.T.dot(direction)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            Beps = bs.compute(points + eps * direction, use_cpp=use_cpp).B[idx]
            deriv_est = (Beps-B0)/(eps)
            new_err = np.linalg.norm(deriv-deriv_est)
            assert new_err < 0.55 * err
            err = new_err

@pytest.mark.parametrize("use_cpp", [True, False])
@pytest.mark.parametrize("idx", [0, 16])
def test_biotsavart_gradient_symmetric_and_divergence_free(use_cpp, idx):
    coil = get_coil()
    bs = BiotSavart([coil], [1e4])
    points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    points += 0.001 * (np.random.rand(*points.shape)-0.5)
    dB = bs.compute(points, use_cpp=use_cpp).dB_by_dX
    assert abs(dB[idx][0, 0] + dB[idx][1, 1] + dB[idx][2, 2]) < 1e-14
    assert np.allclose(dB[idx], dB[idx].T)

@pytest.mark.parametrize("idx", [0, 16])
@pytest.mark.parametrize("use_cpp", [True, False])
def test_dB_by_dcoilcoeff_taylortest(use_cpp, idx):
    coil = get_coil()
    bs = BiotSavart([coil], [1e4])
    points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    points += 0.001 * (np.random.rand(*points.shape)-0.5)

    coil_dofs = coil.get_dofs()
    B0 = bs.compute(points, use_cpp=use_cpp).B[0]

    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dB_dh = h @ bs.compute_by_dcoilcoeff(points, use_cpp=use_cpp).dB_by_dcoilcoeffs[0][0,:,:]
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        Bh = bs.compute(points, use_cpp=use_cpp).B[0]
        deriv_est = (Bh-B0)/eps
        err_new = np.linalg.norm(deriv_est-dB_dh)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

@pytest.mark.parametrize("use_cpp", [True, False])
@pytest.mark.parametrize("idx", [0, 16])
def test_dB_dX_by_dcoilcoeff_taylortest(use_cpp, idx):
    coil = get_coil()
    bs = BiotSavart([coil], [1e4])
    points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    points += 0.001 * (np.random.rand(*points.shape)-0.5)

    coil_dofs = coil.get_dofs()
    dB_dX0 = bs.compute(points, use_cpp=use_cpp).dB_by_dX[idx]

    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dB_dXdh = np.einsum('i,ijk->jk', h, bs.compute_by_dcoilcoeff(points, use_cpp=use_cpp).d2B_by_dXdcoilcoeffs[0][idx,:,:,:])
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        dB_dXh = bs.compute(points, use_cpp=use_cpp).dB_by_dX[idx]
        deriv_est = (dB_dXh-dB_dX0)/eps
        err_new = np.linalg.norm(deriv_est-dB_dXdh)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

@pytest.mark.parametrize("idx", [0, 16])
@pytest.mark.parametrize("use_cpp", [True, False])
def test_d2B_by_dXdX_is_symmetric(use_cpp, idx):
    coil = get_coil()
    bs = BiotSavart([coil], [1e4])
    points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    points += 0.001 * (np.random.rand(*points.shape)-0.5)
    d2B_by_dXdX = bs.compute(points, use_cpp=use_cpp).d2B_by_dXdX
    for i in range(3):
        assert np.allclose(d2B_by_dXdX[idx, :, :, i], d2B_by_dXdX[idx, :, :, i].T)


@pytest.mark.parametrize("idx", [0, 16])
@pytest.mark.parametrize("use_cpp", [True, False])
def test_biotsavart_d2B_by_dXdX_taylortest(use_cpp, idx):
    coil = get_coil()
    bs = BiotSavart([coil], [1e4])
    points = np.asarray(17 *[[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    bs.compute(points, use_cpp=use_cpp)
    B0, dB_by_dX, d2B_by_dXdX = bs.B, bs.dB_by_dX, bs.d2B_by_dXdX
    for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
        first_deriv = dB_by_dX[idx].T.dot(direction)
        second_deriv = np.einsum('ijk,i,j->k', d2B_by_dXdX[idx], direction, direction)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            Beps = bs.compute(points + eps * direction, use_cpp=use_cpp).B[idx]
            deriv_est = (Beps-B0)/(eps)
            second_deriv_est = 2*(deriv_est - first_deriv)/eps
            new_err = np.linalg.norm(second_deriv-second_deriv_est)
            assert new_err < 0.55 * err
            err = new_err


if __name__ == "__main__":
    test_biotsavart_gradient_symmetric_and_divergence_free(True)
    import sys
    sys.exit()
    coil = CartesianFourierCurve(3)
    coil.coefficients[1][0] = 1.
    coil.coefficients[1][1] = 0.5
    coil.coefficients[2][2] = 0.5

    ma = StelleratorSymmetricCylindricalFourierCurve(3, 2)
    ma.coefficients[0][0] = 1.
    ma.coefficients[0][1] = 0.1
    ma.coefficients[1][0] = 0.1

    bs = BiotSavart([coil], [1])
    points = ma.gamma(np.linspace(0., 1., 1000))
    B = bs.B(points)
    dB = bs.dB_by_dX(points)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    coil.plot(plot_derivative=True, show=False, ax=ax)
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.linalg.norm(B, axis=1), cmap=plt.cm.CMRmap)
    # https://stackoverflow.com/questions/15617207/line-colour-of-3d-parametric-curve-in-pythons-matplotlib-pyplot
    fig.colorbar(p)
    plt.show()


