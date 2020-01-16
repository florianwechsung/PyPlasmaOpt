from pyplasmaopt import *
import numpy as np
from math import pi
import argparse

def get_objective():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--at-optimum", dest="at_optimum", default=False,
                        action="store_true")
    parser.add_argument("--mode", type=str, default="deterministic",
                        choices=["deterministic", "stochastic"])
    parser.add_argument("--sigma", type=float, default=3e-3)
    parser.add_argument("--length-scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ppp", type=int, default=20)
    parser.add_argument("--nsamples", type=int, default=100)
    parser.add_argument("--curvature-penalty", type=float, default=0.)
    parser.add_argument("--torsion-penalty", type=float, default=0.)
    parser.add_argument("--tikhonov", type=float, default=0.)
    args, _ = parser.parse_known_args()
    print("Configuration: \n", args.__dict__)

    nfp = 2
    (coils, ma) = get_matt_data(nfp=nfp, ppp=args.ppp, at_optimum=args.at_optimum)
    if args.at_optimum:
        currents = [1e5 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
        eta_bar = -2.105800979374183
    else:
        currents = [0 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
        eta_bar = -2.25
    stellarator = CoilCollection(coils, currents, nfp, True)
    np.random.seed(args.seed)
    outdir = "output-4th-power"
    for k in args.__dict__:
        outdir += "_%s-%s" % (k, args.__dict__[k])
    outdir = outdir.replace(".", "p")
    outdir += "/"
    obj = Problem2_Objective(stellarator, ma, curvature_scale=args.curvature_penalty, torsion_scale=args.torsion_penalty, tikhonov=args.tikhonov, eta_bar=eta_bar, nsamples=args.nsamples, sigma_perturb=args.sigma, length_scale_perturb=args.length_scale, mode=args.mode, outdir=outdir)
    return obj, args

class Problem2_Objective():

    def __init__(self, stellarator, ma, 
                 iota_target=0.103, coil_length_target=4.398229715025710, magnetic_axis_length_target=6.356206812106860,
                 eta_bar=-2.25,
                 curvature_scale=1e-6, torsion_scale=1e-4, tikhonov=0.0,
                 nsamples=0, sigma_perturb=1e-4, length_scale_perturb=0.2, mode="deterministic",
                 outdir="output/"
                 ):
        self.stellarator = stellarator
        self.ma = ma
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        self.biotsavart = bs
        self.biotsavart.set_points(self.ma.gamma)
        qsf = QuasiSymmetricField(eta_bar, ma)
        self.qsf = qsf
        sigma = qsf.sigma
        iota = qsf.iota

        self.J_BSvsQS          = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
        coils = stellarator._base_coils
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
        self.current_fak = 1./(4 * pi * 1e-7)
        self.ma_dof_idxs = (1, 1+self.num_ma_dofs)
        self.current_dof_idxs = (self.ma_dof_idxs[1], self.ma_dof_idxs[1] + len(stellarator.get_currents()))
        self.coil_dof_idxs = (self.current_dof_idxs[1], self.current_dof_idxs[1] + len(stellarator.get_dofs()))
        self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellarator.get_currents()/self.current_fak, self.stellarator.get_dofs()))
        self.x = self.x0.copy()
        self.tikhonov = tikhonov
        self.Jvals = []
        self.dJvals = []

        sampler = GaussianSampler(coils[0].points, length_scale=length_scale_perturb, sigma=sigma_perturb)
        self.sampler = sampler
        self.J_BSvsQS_perturbed = []
        for i in range(nsamples):
            perturbed_coils = [GaussianPerturbedCurve(coil, sampler) for coil in stellarator.coils]
            perturbed_bs    = BiotSavart(perturbed_coils, stellarator.currents)
            perturbed_bs.set_points(self.ma.gamma)
            self.J_BSvsQS_perturbed.append(BiotSavartQuasiSymmetricFieldDifference(self.qsf, perturbed_bs))
        self.Jvals_perturbed = []
        self.Jvals_quantiles = []
        self.Jvals_no_noise = []
        self.xiterates = []
        self.mode = mode
        self.outdir = outdir

    def set_dofs(self, x):
        x_etabar = x[0]
        x_ma = x[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
        x_current = x[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
        x_coil = x[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]

        self.qsf.eta_bar = x_etabar
        self.ma.set_dofs(x_ma)
        self.biotsavart.set_points(self.ma.gamma)
        self.stellarator.set_currents(self.current_fak * x_current)
        self.stellarator.set_dofs(x_coil)

    def update(self, x):
        self.x[:] = x
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
        bs = self.biotsavart

        self.set_dofs(x)

        bs.clear_cached_properties()
        qsf.clear_cached_properties()

        self.dresetabar  = np.zeros(1)
        self.dresma      = np.zeros(self.ma_dof_idxs[1]-self.ma_dof_idxs[0])
        self.drescurrent = np.zeros(self.current_dof_idxs[1]-self.current_dof_idxs[0])
        self.drescoil    = np.zeros(self.coil_dof_idxs[1]-self.coil_dof_idxs[0])


        """ Objective values """
        ninv = 1./len(self.J_BSvsQS_perturbed)

        for j in self.J_BSvsQS_perturbed:
            j.biotsavart.clear_cached_properties()
            j.biotsavart.set_points(self.ma.gamma)
        if self.mode == "deterministic":
            self.res1         = 0.5 * J_BSvsQS.J_L2() + 0.5 * J_BSvsQS.J_H1()
            self.dresetabar  += 0.5 * J_BSvsQS.dJ_L2_by_detabar() + 0.5 * J_BSvsQS.dJ_H1_by_detabar()
            self.dresma      += 0.5 * J_BSvsQS.dJ_L2_by_dmagneticaxiscoefficients() + 0.5 * J_BSvsQS.dJ_H1_by_dmagneticaxiscoefficients()
            self.drescoil    += 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_L2_by_dcoilcoefficients()) \
                + 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_H1_by_dcoilcoefficients())
            self.drescurrent += 0.5 * self.current_fak * (
                self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_L2_by_dcoilcurrents()) + self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_H1_by_dcoilcurrents())
            )
        elif self.mode == "stochastic":
            self.res1         = ninv * 0.5 * sum(J.J_L2() + J.J_H1() for J in self.J_BSvsQS_perturbed)
            self.dresetabar  += ninv * 0.5 * sum(J.dJ_L2_by_detabar() + J.dJ_H1_by_detabar() for J in self.J_BSvsQS_perturbed)
            self.dresma      += ninv * 0.5 * sum(J.dJ_L2_by_dmagneticaxiscoefficients() + J.dJ_H1_by_dmagneticaxiscoefficients() for J in self.J_BSvsQS_perturbed)
            self.drescoil    += ninv * 0.5 * sum(self.stellarator.reduce_coefficient_derivatives(J.dJ_L2_by_dcoilcoefficients()) \
                + self.stellarator.reduce_coefficient_derivatives(J.dJ_H1_by_dcoilcoefficients())  for J in self.J_BSvsQS_perturbed)
            self.drescurrent += ninv * 0.5 * self.current_fak * sum(self.stellarator.reduce_current_derivatives(J.dJ_L2_by_dcoilcurrents()) \
                                                                    + self.stellarator.reduce_current_derivatives(J.dJ_H1_by_dcoilcurrents()) for J in self.J_BSvsQS_perturbed )
        else:
            raise NotImplementedError

        self.res2      = 0.5 * sum( (1/coil_length_target)**2 * (J2.J() - coil_length_target)**2 for J2 in J_coil_lengths)
        self.drescoil += (1/coil_length_target)**2 * self.stellarator.reduce_coefficient_derivatives([
            (J_coil_lengths[i].J()-coil_length_target) * J_coil_lengths[i].dJ_by_dcoefficients() for i in range(len(J_coil_lengths))])

        self.res3    = 0.5 * (1/magnetic_axis_length_target)**2 * (J_axis_length.J() - magnetic_axis_length_target)**2
        self.dresma += (1/magnetic_axis_length_target)**2 * (J_axis_length.J()-magnetic_axis_length_target) * J_axis_length.dJ_by_dcoefficients()

        self.res4        = 0.5 * (1/iota_target**2) * (qsf.iota-iota_target)**2
        self.dresetabar += (1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_detabar[:,0]
        self.dresma     += (1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_dcoeffs[:, 0]

        if curvature_scale > 1e-30:
            self.res5      = sum(curvature_scale * J.J() for J in J_coil_curvatures)
            self.drescoil += self.curvature_scale * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_curvatures])
        else:
            self.res5 = 0
        if torsion_scale > 1e-30:
            self.res6      = sum(torsion_scale * J.J() for J in J_coil_torsions)
            self.drescoil += self.torsion_scale * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_torsions])
        else:
            self.res6 = 0

        self.res_tikhonov = self.tikhonov * np.sum((x-self.x0)**2)


        self.dres_tikhonov = self.tikhonov * 2. * (x-self.x0)
        self.dresetabar += self.dres_tikhonov[0:1]
        self.dresma += self.dres_tikhonov[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
        self.drescurrent += self.dres_tikhonov[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
        self.drescoil += self.dres_tikhonov[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]

        self.res = self.res1 + self.res2 + self.res3 + self.res4 + self.res5 + self.res6  + self.res_tikhonov

        self.dres = np.concatenate((
            self.dresetabar, self.dresma,
            self.drescurrent, self.drescoil
        ))

        self.perturbed_vals = [self.res - self.res1 + 0.5 * j.J_L2() + 0.5 * j.J_H1() for j in self.J_BSvsQS_perturbed]

    def callback(self, x):
        assert np.allclose(self.x, x)
        self.Jvals.append(self.res)
        norm = np.linalg.norm
        self.dJvals.append((
            norm(self.dres), norm(self.dresetabar), norm(self.dresma), norm(self.drescurrent), norm(self.drescoil)
        ))
        self.Jvals_quantiles.append((np.quantile(self.perturbed_vals, 0.1), np.mean(self.perturbed_vals), np.quantile(self.perturbed_vals, 0.9)))
        self.Jvals_no_noise.append(self.res - self.res1 + 0.5 * (self.J_BSvsQS.J_L2() + self.J_BSvsQS.J_H1()))
        self.xiterates.append(x.copy())
        self.Jvals_perturbed.append(self.perturbed_vals)

        print("################################################################################")
        iteration = len(self.xiterates)-1
        print(f"Iteration {iteration}")
        norm = np.linalg.norm
        print("Objective value:         ", self.res)
        print("Objective values:        ", self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res_tikhonov)
        print("Objective 10%, mean, 90%:", np.quantile(self.perturbed_vals, 0.1), np.mean(self.perturbed_vals), np.quantile(self.perturbed_vals, 0.9))
        print("Objective gradients:     ",
                norm(self.dresetabar),
                norm(self.dresma),
                norm(self.drescurrent),
                norm(self.drescoil))

        max_curvature  = max(np.max(c.kappa) for c in self.stellarator._base_coils)
        mean_curvature = np.mean([np.mean(c.kappa) for c in self.stellarator._base_coils])
        max_torsion    = max(np.max(np.abs(c.torsion)) for c in self.stellarator._base_coils)
        mean_torsion   = np.mean([np.mean(np.abs(c.torsion)) for c in self.stellarator._base_coils])
        print("Curvature Max: %.3e; Mean: %.3e " % (max_curvature, mean_curvature))
        print("Torsion   Max: %.3e; Mean: %.3e " % (max_torsion, mean_torsion), flush=True)
        if iteration % 10 == 0:
            self.plot('iteration-%04i.png' % iteration)

    def plot(self, filename):
        import matplotlib.pyplot as plt
        ax = None
        for i in range(0, len(self.stellarator.coils)):
            ax = self.stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(self.stellarator._base_coils)])
        self.ma.plot(ax=ax, show=False, closed_loop=False)
        ax.view_init(elev=90., azim=0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 1)
        plt.savefig(self.outdir + filename, dpi=300)
        plt.close()
