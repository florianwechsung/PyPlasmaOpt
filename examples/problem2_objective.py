from pyplasmaopt import *

class Problem2_Objective():

    def __init__(self, stellerator, ma, 
                 iota_target=0.103, coil_length_target=4.398229715025710, magnetic_axis_length_target=6.356206812106860,
                 eta_bar=-2.105800979374183,
                 curvature_scale=1e-6, torsion_scale=1e-4, tikhonov=0.0
                 ):
        self.stellerator = stellerator
        self.ma = ma
        bs = BiotSavart(stellerator.coils, stellerator.currents)
        qsf = QuasiSymmetricField(eta_bar, ma)
        self.qsf = qsf
        sigma, iota = qsf.solve_state()

        self.J_BSvsQS          = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
        self.J_BSvsQS.update()
        coils = stellerator._base_coils
        self.J_coil_lengths    = [CurveLength(coil) for coil in coils]
        self.J_axis_length     = CurveLength(ma)
        self.J_coil_curvatures = [CurveCurvature(coil) for coil in coils]
        self.J_coil_torsions   = [CurveTorsion(coil) for coil in coils]

        self.iota_target                 = iota_target
        self.coil_length_target          = coil_length_target
        self.magnetic_axis_length_target = magnetic_axis_length_target
        self.curvature_scale             = curvature_scale
        self.torsion_scale               = torsion_scale
        self.num_ma_dofs = len(ma.get_dofs())
        self.current_fak = 7.957747154594768e+05
        self.ma_dof_idxs = (1, 1+self.num_ma_dofs)
        self.current_dof_idxs = (self.ma_dof_idxs[1], self.ma_dof_idxs[1] + len(stellerator.get_currents()))
        self.coil_dof_idxs = (self.current_dof_idxs[1], self.current_dof_idxs[1] + len(stellerator.get_dofs()))
        self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellerator.get_currents()/self.current_fak, self.stellerator.get_dofs()))
        self.tikhonov = tikhonov

    def update(self, x):
        J_BSvsQS          = self.J_BSvsQS
        J_coil_lengths    = self.J_coil_lengths
        J_axis_length     = self.J_axis_length
        J_coil_curvatures = self.J_coil_curvatures
        J_coil_torsions   = self.J_coil_torsions
        iota_target                 = self.iota_target
        coil_length_target          = self.coil_length_target
        magnetic_axis_length_target = self.magnetic_axis_length_target
        curvature_scale             = self.curvature_scale
        torsion_scale               = self.torsion_scale
        qsf = self.qsf

        x_etabar = x[0]
        x_ma = x[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
        x_current = x[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
        x_coil = x[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
        self.qsf.eta_bar = x_etabar
        self.ma.set_dofs(x_ma)
        self.stellerator.set_currents(self.current_fak * x_current)
        self.stellerator.set_dofs(x_coil)
        self.qsf.solve_state()
        self.J_BSvsQS.update()
        self.res1 = 0.5 * J_BSvsQS.J_L2() + 0.5 * J_BSvsQS.J_H1()
        self.res2 = 0.5 * sum( (1/coil_length_target)**2 * (J2.J() - coil_length_target)**2 for J2 in J_coil_lengths)
        self.res3 = 0.5 * (1/magnetic_axis_length_target)**2 * (J_axis_length.J() - magnetic_axis_length_target)**2
        self.res4 = 0.5 * (1/iota_target**2) * (qsf.state[-1]-iota_target)**2
        self.res5 = sum(curvature_scale * J.J() for J in J_coil_curvatures)
        self.res6 = sum(torsion_scale * J.J() for J in J_coil_torsions)
        self.res_tikhonov = self.tikhonov * np.sum((x-self.x0)**2)

        self.dres1etabar = 0.5 * J_BSvsQS.dJ_L2_by_detabar() + 0.5 * J_BSvsQS.dJ_H1_by_detabar()
        self.dres4etabar = (1/iota_target**2) * (qsf.state[-1] - iota_target) * J_BSvsQS.diota_by_detabar[:,0]
        self.dres1coil = 0.5 * self.stellerator.reduce_coefficient_derivatives(J_BSvsQS.dJ_L2_by_dcoilcoefficients()) \
            + 0.5 * self.stellerator.reduce_coefficient_derivatives(J_BSvsQS.dJ_H1_by_dcoilcoefficients())
        self.dres2coil = (1/coil_length_target)**2 * self.stellerator.reduce_coefficient_derivatives([
            (J_coil_lengths[i].J()-coil_length_target) * J_coil_lengths[i].dJ_by_dcoefficients() for i in range(len(J_coil_lengths))])
        self.dres5coil = self.curvature_scale * self.stellerator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_curvatures])
        self.dres6coil = self.torsion_scale * self.stellerator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_torsions])

        self.dres1current = 0.5 * self.current_fak * (
            self.stellerator.reduce_current_derivatives(J_BSvsQS.dJ_L2_by_dcoilcurrents()) + self.stellerator.reduce_current_derivatives(J_BSvsQS.dJ_H1_by_dcoilcurrents())
        )
        self.dres1ma = 0.5 * J_BSvsQS.dJ_L2_by_dmagneticaxiscoefficients() + 0.5 * J_BSvsQS.dJ_H1_by_dmagneticaxiscoefficients()
        self.dres3ma = (1/magnetic_axis_length_target)**2 * (J_axis_length.J()-magnetic_axis_length_target) * J_axis_length.dJ_by_dcoefficients()
        self.dres4ma = (1/iota_target**2) * (qsf.state[-1] - iota_target) * J_BSvsQS.diota_by_dcoeffs[:,0]

        self.dres_tikhonov = self.tikhonov * 2. * (x-self.x0)

        self.res = self.res1 + self.res2 + self.res3 + self.res4 + self.res5 + self.res6  + self.res_tikhonov
        self.dres = np.concatenate((self.dres1etabar + self.dres4etabar, self.dres1ma + self.dres3ma + self.dres4ma, self.dres1current, self.dres1coil + self.dres2coil + self.dres5coil + self.dres6coil))
        self.dres += self.dres_tikhonov

    def print_status(self):
        print("Objective values:", self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res_tikhonov)
        print("Objective gradients:", np.linalg.norm(self.dres1etabar + self.dres4etabar), np.linalg.norm(self.dres1ma + self.dres3ma + self.dres4ma + self.dres_tikhonov[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]), np.linalg.norm(self.dres1current + self.dres_tikhonov[self.current_dof_idxs[0]:self.current_dof_idxs[1]]), np.linalg.norm(self.dres1coil + self.dres2coil + self.dres5coil + self.dres6coil + self.dres_tikhonov[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]))


