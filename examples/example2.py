from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve, \
    BiotSavartQuasiSymmetricFieldDifference, get_matt_data, CoilCollection, \
    CurveLength, CurveCurvature, QuasiSymmetricField
import numpy as np

nfp = 2
(coils, ma) = get_matt_data(nfp=nfp, ppp=20)
currents = [1e5 * x for x in [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
stellerator = CoilCollection(coils, currents, nfp, True)
bs = BiotSavart(stellerator.coils, stellerator.currents)
eta_bar = -2.105800979374183
qsf = QuasiSymmetricField(eta_bar, ma)
sigma, iota = qsf.solve_state()
J1 = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
J1.update()

J2s = [CurveLength(coil) for coil in coils]
coil_length_target = 4.398229715025710
J3s = [CurveCurvature(coil) for coil in coils]

coil_dofs = stellerator.get_dofs()
num_ma_dofs = len(ma.get_dofs())
iota_target = 0.103
def J(x, info=None):
    stellerator.set_dofs(x[num_ma_dofs:])
    ma.set_dofs(x[0:num_ma_dofs])
    qsf.solve_state()
    J1.update()
    res1 = J1.J_L2() + J1.J_H1()
    res2s = [J2.J() for J2 in J2s]
    res2 = sum( (1/coil_length_target)**2 * (j - coil_length_target)**2 for j in res2s)
    res3 = sum(J3.J() for J3 in J3s)
    res4 = (1/iota_target**2) * (qsf.state[-1]-iota_target)**2

    dres1coil = stellerator.reduce_derivatives(J1.dJ_L2_by_dcoilcoefficients()) + stellerator.reduce_derivatives(J1.dJ_H1_by_dcoilcoefficients())
    dres2coil = stellerator.reduce_derivatives([2 * (1/coil_length_target)**2 * (res2s[i]-coil_length_target) * J2s[i].dJ_by_dcoefficients() for i in range(len(J2s))])
    dres3coil = stellerator.reduce_derivatives([J3.dJ_by_dcoefficients() for J3 in J3s])

    dres1ma = J1.dJ_L2_by_dmagneticaxiscoefficients() + J1.dJ_H1_by_dmagneticaxiscoefficients()
    dres4ma = 2 * (1/iota_target**2) * (qsf.state[-1] - iota_target) * J1.diota_by_dcoeffs[:,0]

    # if info is not None:
    if False:
        info['Nfeval'] += 1
        if info['Nfeval'] % 10 == 0:
            ax = None
            for i in range(0, len(coils)):
                ax = coils[i].plot(ax=ax, show=False)
            ma.plot(ax=ax, show=False)
            ax.view_init(elev=10., azim=0)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1, 1)
            import matplotlib.pyplot as plt
            plt.savefig('output/%i.png' % info['Nfeval'])
            plt.close()
    dres = np.concatenate((dres1ma + dres4ma, dres1coil + dres2coil + dres3coil))
    return res1 + res2 + res3 + res4, dres
x = np.concatenate((ma.get_dofs(), stellerator.get_dofs()))
# J0, dJ0 = J(x)
# np.random.seed(1)
# h = 1e-2 * np.random.rand(*(x.shape))
# dJh = sum(dJ0*h)
# for i in range(5, 15):
#     eps = 0.5**i
#     Jeps = J(x + eps * h)[0]
#     err = abs((Jeps-J0)/eps - dJh)
#     print(err)

import time
from scipy.optimize import minimize
maxiter = 100
t1 = time.time()
res = minimize(J, x, args=({'Nfeval':0},), jac=True, method='L-BFGS-B', tol=1e-20, options={'maxiter': maxiter})
t2 = time.time()
print(f"Time per iteration: {(t2-t1)/maxiter:.4f}")
coil_dofs = res.x
print("Gradient norm at minimum:", np.linalg.norm(res.jac))

# J0, dJ0 = J(coil_dofs)
# h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
# dJh = sum(dJ0*h)
# for i in range(5, 10):
#     eps = 0.5**i
#     Jeps = J(coil_dofs + eps * h)[0]
#     err = abs((Jeps-J0)/eps - dJh)
#     print(err)
