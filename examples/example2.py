from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve, \
    BiotSavartQuasiSymmetricFieldDifference, get_matt_data, CoilCollection, \
    CurveLength, CurveCurvature, QuasiSymmetricField, CurveTorsion
import numpy as np
import matplotlib.pyplot as plt

nfp = 2
(coils, ma) = get_matt_data(nfp=nfp, ppp=20, at_optimum=True)
currents = [1e5 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
stellerator = CoilCollection(coils, currents, nfp, True)
bs = BiotSavart(stellerator.coils, stellerator.currents)
eta_bar = -2.105800979374183
qsf = QuasiSymmetricField(eta_bar, ma)
sigma, iota = qsf.solve_state()

J_BSvsQS          = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
J_BSvsQS.update()
J_coil_lengths    = [CurveLength(coil) for coil in coils]
J_axis_length     = CurveLength(ma)
J_coil_curvatures = [CurveCurvature(coil) for coil in coils]
J_coil_torsions   = [CurveTorsion(coil) for coil in coils]
curvature_scale = 1e-6
torsion_scale   = 1e-4

coil_length_target = 4.398229715025710
magnetic_axis_length_target = 6.356206812106860


num_ma_dofs = len(ma.get_dofs())
iota_target = 0.103
current_fak = 7.957747154594768e+05


def J(x, info=None):
    ma.set_dofs(x[0:num_ma_dofs])
    stellerator.set_currents(current_fak * x[num_ma_dofs:num_ma_dofs+len(coils)])
    stellerator.set_dofs(x[num_ma_dofs+len(coils):])
    qsf.solve_state()
    J_BSvsQS.update()
    res1 = J_BSvsQS.J_L2() + J_BSvsQS.J_H1()
    res2 = sum( (1/coil_length_target)**2 * (J2.J() - coil_length_target)**2 for J2 in J_coil_lengths)
    res3 = (1/magnetic_axis_length_target)**2 * (J_axis_length.J() - magnetic_axis_length_target)**2
    res4 = (1/iota_target**2) * (qsf.state[-1]-iota_target)**2
    res5 = sum(curvature_scale * J.J() for J in J_coil_curvatures)
    res6 = sum(torsion_scale * J.J() for J in J_coil_torsions)

    dres1coil = stellerator.reduce_coefficient_derivatives(J_BSvsQS.dJ_L2_by_dcoilcoefficients()) + stellerator.reduce_coefficient_derivatives(J_BSvsQS.dJ_H1_by_dcoilcoefficients())
    dres2coil = stellerator.reduce_coefficient_derivatives([
        2 * (1/coil_length_target)**2 * (J_coil_lengths[i].J()-coil_length_target) * J_coil_lengths[i].dJ_by_dcoefficients() for i in range(len(J_coil_lengths))
    ])
    dres5coil = curvature_scale * stellerator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_curvatures])
    dres6coil = torsion_scale * stellerator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_torsions])

    dres1current = current_fak * (
        stellerator.reduce_current_derivatives(J_BSvsQS.dJ_L2_by_dcoilcurrents()) + stellerator.reduce_current_derivatives(J_BSvsQS.dJ_H1_by_dcoilcurrents())
    )
    dres1ma = J_BSvsQS.dJ_L2_by_dmagneticaxiscoefficients() + J_BSvsQS.dJ_H1_by_dmagneticaxiscoefficients()
    dres3ma = 2 * (1/magnetic_axis_length_target)**2 * (J_axis_length.J()-magnetic_axis_length_target) * J_axis_length.dJ_by_dcoefficients()
    dres4ma = 2 * (1/iota_target**2) * (qsf.state[-1] - iota_target) * J_BSvsQS.diota_by_dcoeffs[:,0]

    res = res1 + res2 + res3 + res4 + res5 + res6
    dres = np.concatenate((dres1ma + dres3ma + dres4ma, dres1current, dres1coil + dres2coil + dres5coil + dres6coil))
    # if False:
    if info is not None:
        if info['Nfeval'] % 10 == 0:
            ax = None
            for i in range(0, len(stellerator.coils)):
                ax = stellerator.coils[i].plot(ax=ax, show=False)
            ma.plot(ax=ax, show=False, closed_loop=False)
            ax.view_init(elev=30., azim=0)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1, 1)
            plt.savefig('output/%i.png' % info['Nfeval'])
            plt.close()
            print("################################################################################")
            print(f"Iteration {info['Nfeval']}")
            print("Objective values:", res1, res2, res3, res4, res5, res6)
            print("Objective gradients:", np.linalg.norm(dres1ma + dres3ma + dres4ma), np.linalg.norm(dres1current), np.linalg.norm(dres1coil + dres2coil + dres5coil + dres6coil))
        info['Nfeval'] += 1
    return res, dres
x = np.concatenate((ma.get_dofs(), stellerator.get_currents()/current_fak, stellerator.get_dofs()))
if False:
    J0, dJ0 = J(x)
    np.random.seed(1)
    h = 1e-3 * np.random.rand(*(x.shape))
    dJh = sum(dJ0*h)
    for i in range(5, 20):
        eps = 0.5**i
        Jeps = J(x + eps * h)[0]
        err = abs((Jeps-J0)/eps - dJh)
        print(err)
    import sys
    sys.exit()

import time
from scipy.optimize import minimize
maxiter = 100
t1 = time.time()
args = {'Nfeval':0}
res = minimize(J, x, args=(args,), jac=True, method='BFGS', tol=1e-20, options={'maxiter': maxiter})
t2 = time.time()
print(f"Time per iteration: {(t2-t1)/args['Nfeval']:.4f}")
x = res.x
print("Gradient norm at minimum:", np.linalg.norm(res.jac), np.linalg.norm(J(x)[1]))

# J0, dJ0 = J(coil_dofs)
# h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
# dJh = sum(dJ0*h)
# for i in range(5, 10):
#     eps = 0.5**i
#     Jeps = J(coil_dofs + eps * h)[0]
#     err = abs((Jeps-J0)/eps - dJh)
#     print(err)
# import IPython; IPython.embed()
# ax = None
# for i in range(0, len(stellerator.coils)):
#     ax = stellerator.coils[i].plot(ax=ax, show=False)
# ma.plot(ax=ax, show=False, closed_loop=False)
# ax.view_init(elev=30., azim=45)
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-1, 1)
# plt.show()
# import sys
# sys.exit()
