import numpy as np
import pytest
from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve

def get_coil():
    coil = CartesianFourierCurve(3)
    coil.coefficients[1][0] = 1.
    coil.coefficients[1][1] = 0.5
    coil.coefficients[2][2] = 0.5
    return coil

def test_biotsavart_exponential_convergence():
    coil = get_coil()
    points = np.asarray([[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    btrue   = BiotSavart([coil], [1e4], 1e3).B(points)
    bcoarse = BiotSavart([coil], [1e4], 10 ).B(points)
    bfine   = BiotSavart([coil], [1e4], 20 ).B(points)
    assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)

    dbtrue   = BiotSavart([coil], [1e4], 1e3).dB_by_dX(points)
    dbcoarse = BiotSavart([coil], [1e4], 10 ).dB_by_dX(points)
    dbfine   = BiotSavart([coil], [1e4], 20 ).dB_by_dX(points)
    assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)

def test_biotsavart_dBdX_taylortest():
    coil = get_coil()
    bs = BiotSavart([coil], [1e4], 100)
    points = np.asarray([[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    dB = bs.dB_by_dX(points)
    B0 = bs.B(points)
    for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
        deriv = dB[0].dot(direction)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            Beps = bs.B(points + eps * direction)
            deriv_est = (Beps-B0)/(eps)
            new_err = np.linalg.norm(deriv-deriv_est)
            assert new_err < 0.55 * err
            err = new_err

def test_biotsavart_gradient_symmetric_and_divergence_free():
    coil = get_coil()
    bs = BiotSavart([coil], [1e4], 100)
    points = np.asarray([[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    dB = bs.dB_by_dX(points)
    assert abs(dB[0][0, 0] + dB[0][1, 1] + dB[0][2, 2]) < 1e-14
    assert np.allclose(dB[0], dB[0].T)

if __name__ == "__main__":
    coil = CartesianFourierCurve(3)
    coil.coefficients[1][0] = 1.
    coil.coefficients[1][1] = 0.5
    coil.coefficients[2][2] = 0.5

    ma = StelleratorSymmetricCylindricalFourierCurve(3, 2)
    ma.coefficients[0][0] = 1.
    ma.coefficients[0][1] = 0.1
    ma.coefficients[1][0] = 0.1

    bs = BiotSavart([coil], [1], 100)
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


