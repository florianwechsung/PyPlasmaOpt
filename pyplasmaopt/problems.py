from simsgeo import BiotSavart
from simsgeo import MinimumDistance as sgMinimumDistance
from .quasi_symmetric_field import QuasiSymmetricField
from .objective import BiotSavartQuasiSymmetricFieldDifference, CurveLength, CurveTorsion, CurveCurvature, SobolevTikhonov, UniformArclength, MinimumDistance, CoilLpReduction
from .curve import GaussianSampler, UniformSampler
from .stochastic_objective import StochasticQuasiSymmetryObjective, CVaR
from .logging import info

from mpi4py import MPI
comm = MPI.COMM_WORLD
from math import pi, sin, cos
import numpy as np
import os

class NearAxisQuasiSymmetryObjective():

    def __init__(self, stellarator, ma, iota_target, eta_bar=-2.25,
                 coil_length_target=None, magnetic_axis_length_target=None,
                 curvature_weight=0.0, torsion_weight=0., tikhonov_weight=0., arclength_weight=0., sobolev_weight=0.,
                 minimum_distance=0.04, distance_weight=0.,
                 ninsamples=0, noutsamples=0, sigma_perturb=1e-4, length_scale_perturb=0.2, mode="deterministic",
                 outdir="output/", seed=1, freq_plot=250, freq_out_of_sample=1, distribution='gaussian',
                 ):
        self.stellarator = stellarator
        self.seed = seed
        self.ma = ma
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        self.biotsavart = bs
        self.biotsavart.set_points(self.ma.gamma())
        qsf = QuasiSymmetricField(eta_bar, ma)
        self.qsf = qsf
        sigma = qsf.sigma
        iota = qsf.iota
        self.ninsamples = ninsamples
        self.noutsamples = noutsamples

        self.J_BSvsQS          = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
        coils = stellarator._base_coils
        self.J_coil_lengths    = [CurveLength(coil) for coil in coils]
        self.J_axis_length     = CurveLength(ma)
        if coil_length_target is not None:
            self.coil_length_targets = [coil_length_target for coil in coils]
        else:
            self.coil_length_targets = [J.J() for J in self.J_coil_lengths]
        self.magnetic_axis_length_target = magnetic_axis_length_target or self.J_axis_length.J()

        self.J_coil_curvatures = [CurveCurvature(coil, length) for (coil, length) in zip(coils, self.coil_length_targets)]
        self.J_coil_torsions   = [CurveTorsion(coil, p=2) for coil in coils]
        self.J_sobolev_weights = [SobolevTikhonov(coil, weights=[1., .1, .1, .1]) for coil in coils] + [SobolevTikhonov(ma, weights=[1., .1, .1, .1])]
        self.J_arclength_weights = [UniformArclength(coil, length) for (coil, length) in zip(coils, self.coil_length_targets)]
        self.J_distance = MinimumDistance(stellarator.coils, minimum_distance)
        # self.J_distance = sgMinimumDistance(stellarator.coils, minimum_distance)

        self.iota_target                 = iota_target
        self.curvature_weight             = curvature_weight
        self.torsion_weight               = torsion_weight
        self.num_ma_dofs = len(ma.get_dofs())
        self.current_fak = 1./(4 * pi * 1e-7)
        self.ma_dof_idxs = (1, 1+self.num_ma_dofs)
        self.current_dof_idxs = (self.ma_dof_idxs[1], self.ma_dof_idxs[1] + len(stellarator.get_currents()))
        self.coil_dof_idxs = (self.current_dof_idxs[1], self.current_dof_idxs[1] + len(stellarator.get_dofs()))
        if mode in ["deterministic", "stochastic"]:
            self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellarator.get_currents()/self.current_fak, self.stellarator.get_dofs()))
        elif mode[0:4] == "cvar":
            self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellarator.get_currents()/self.current_fak, self.stellarator.get_dofs(), [0.]))
        else:
            raise NotImplementedError
        self.x = self.x0.copy()
        self.sobolev_weight = sobolev_weight
        self.tikhonov_weight = tikhonov_weight
        self.arclength_weight = arclength_weight
        self.distance_weight = distance_weight
        self.freq_plot = freq_plot
        self.freq_out_of_sample = freq_out_of_sample

        if distribution == 'gaussian':
            sampler = GaussianSampler(coils[0].quadpoints, length_scale=length_scale_perturb, sigma=sigma_perturb)
        elif distribution == 'uniform':
            sampler = UniformSampler(coils[0].quadpoints, length_scale=length_scale_perturb, sigma=sigma_perturb)
        else:
            raise NotImplementedError
        # import IPython; IPython.embed()
        # import sys; sys.exit()
        self.sampler = sampler

        self.stochastic_qs_objective = StochasticQuasiSymmetryObjective(stellarator, sampler, ninsamples, qsf, self.seed)
        self.stochastic_qs_objective_out_of_sample = None

        if mode in ["deterministic", "stochastic"]:
            self.mode = mode
        elif mode[0:4] == "cvar":
            self.mode = "cvar"
            self.cvar_alpha = float(mode[4:])
            self.cvar = CVaR(self.cvar_alpha, .01)
        else:
            raise NotImplementedError

        self.stochastic_qs_objective.set_magnetic_axis(self.ma.gamma())

        self.Jvals_perturbed = []
        self.Jvals_quantiles = []
        self.Jvals_no_noise = []
        self.xiterates = []
        self.Jvals_individual = []
        self.QSvsBS_perturbed = []
        self.Jvals = []
        self.dJvals = []
        self.out_of_sample_values = []
        self.outdir = outdir

    def set_dofs(self, x):
        x_etabar = x[0]
        x_ma = x[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
        x_current = x[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
        x_coil = x[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
        self.t = x[-1]

        self.qsf.eta_bar = x_etabar
        self.ma.set_dofs(x_ma)
        self.biotsavart.set_points(self.ma.gamma())
        self.stellarator.set_currents(self.current_fak * x_current)
        self.stellarator.set_dofs(x_coil)

        self.biotsavart.clear_cached_properties()
        self.qsf.clear_cached_properties()
        self.J_BSvsQS.clear_cached_properties()
        for J in self.stochastic_qs_objective.J_BSvsQS_perturbed:
            J.clear_cached_properties()

    def update(self, x):
        self.x[:] = x
        J_BSvsQS          = self.J_BSvsQS
        J_coil_lengths    = self.J_coil_lengths
        J_axis_length     = self.J_axis_length
        J_coil_curvatures = self.J_coil_curvatures
        J_coil_torsions   = self.J_coil_torsions

        iota_target                 = self.iota_target
        magnetic_axis_length_target = self.magnetic_axis_length_target
        curvature_weight             = self.curvature_weight
        torsion_weight               = self.torsion_weight
        qsf = self.qsf

        self.set_dofs(x)

        self.dresetabar  = np.zeros(1)
        self.dresma      = np.zeros(self.ma_dof_idxs[1]-self.ma_dof_idxs[0])
        self.drescurrent = np.zeros(self.current_dof_idxs[1]-self.current_dof_idxs[0])
        self.drescoil    = np.zeros(self.coil_dof_idxs[1]-self.coil_dof_idxs[0])


        """ Objective values """

        self.res2      = 0.5 * sum( (1/l)**2 * (J2.J() - l)**2 for (J2, l) in zip(J_coil_lengths, self.coil_length_targets))
        self.drescoil += self.stellarator.reduce_coefficient_derivatives([
            (1/l)**2 * (J_coil_lengths[i].J()-l) * J_coil_lengths[i].dJ_by_dcoefficients() for (i, l) in zip(list(range(len(J_coil_lengths))), self.coil_length_targets)])

        self.res3    = 0.5 * (1/magnetic_axis_length_target)**2 * (J_axis_length.J() - magnetic_axis_length_target)**2
        self.dresma += (1/magnetic_axis_length_target)**2 * (J_axis_length.J()-magnetic_axis_length_target) * J_axis_length.dJ_by_dcoefficients()

        self.res4        = 0.5 * (1/iota_target**2) * (qsf.iota-iota_target)**2
        self.dresetabar += (1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_detabar[:,0]
        self.dresma     += (1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_dcoeffs[:, 0]

        if curvature_weight > 1e-15:
            self.res5      = sum(curvature_weight * J.J() for J in J_coil_curvatures)
            self.drescoil += self.curvature_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_curvatures])
        else:
            self.res5 = 0
        if torsion_weight > 1e-15:
            self.res6      = sum(torsion_weight * J.J() for J in J_coil_torsions)
            self.drescoil += self.torsion_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_torsions])
        else:
            self.res6 = 0

        if self.sobolev_weight > 1e-15:
            self.res7 = sum(self.sobolev_weight * J.J() for J in self.J_sobolev_weights)
            self.drescoil += self.sobolev_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_sobolev_weights[:-1]])
            self.dresma += self.sobolev_weight * self.J_sobolev_weights[-1].dJ_by_dcoefficients()
        else:
            self.res7 = 0

        if self.arclength_weight > 1e-15:
            self.res8 = sum(self.arclength_weight * J.J() for J in self.J_arclength_weights)
            self.drescoil += self.arclength_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_arclength_weights])
        else:
            self.res8 = 0

        if self.distance_weight > 1e-15:
            self.res9 = self.distance_weight * self.J_distance.J()
            self.drescoil += self.distance_weight * self.stellarator.reduce_coefficient_derivatives(self.J_distance.dJ_by_dcoefficients())
            # self.drescoil += self.distance_weight * self.stellarator.reduce_coefficient_derivatives(self.J_distance.dJ())
        else:
            self.res9 = 0

        if self.tikhonov_weight > 1e-15:
            self.res_tikhonov_weight = self.tikhonov_weight * np.sum((x-self.x0)**2)
            dres_tikhonov_weight = self.tikhonov_weight * 2. * (x-self.x0)
            self.dresetabar += dres_tikhonov_weight[0:1]
            self.dresma += dres_tikhonov_weight[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
            self.drescurrent += dres_tikhonov_weight[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
            self.drescoil += dres_tikhonov_weight[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
        else:
            self.res_tikhonov_weight = 0

        self.stochastic_qs_objective.set_magnetic_axis(self.ma.gamma())

        Jsamples = self.stochastic_qs_objective.J_samples()
        self.Jsamples = Jsamples
        assert len(Jsamples) == self.ninsamples
        if comm.rank == 0:
            self.QSvsBS_perturbed.append(Jsamples)

        self.res1_det        = 0.5 * J_BSvsQS.J_L2() + 0.5 * J_BSvsQS.J_H1()
        self.dresetabar_det  = 0.5 * J_BSvsQS.dJ_L2_by_detabar() + 0.5 * J_BSvsQS.dJ_H1_by_detabar()
        self.dresma_det      = 0.5 * J_BSvsQS.dJ_L2_by_dmagneticaxiscoefficients() + 0.5 * J_BSvsQS.dJ_H1_by_dmagneticaxiscoefficients()
        self.drescoil_det    = 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_L2_by_dcoilcoefficients) \
            + 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_H1_by_dcoilcoefficients)
        self.drescurrent_det = 0.5 * self.current_fak * (
            self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_L2_by_dcoilcurrents()) + self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_H1_by_dcoilcurrents())
        )
        if self.mode == "deterministic":
            self.res1         = self.res1_det
            self.dresetabar  += self.dresetabar_det
            self.dresma      += self.dresma_det
            self.drescoil    += self.drescoil_det
            self.drescurrent += self.drescurrent_det
        else:
            self.dresetabar_det  += self.dresetabar
            self.dresma_det      += self.dresma
            self.drescoil_det    += self.drescoil
            self.drescurrent_det += self.drescurrent
            if self.mode == "stochastic":
                n = self.ninsamples
                self.res1         = sum(Jsamples)/n
                self.res1_det     = self.res1
                self.drescoil    += sum(self.stochastic_qs_objective.dJ_by_dcoilcoefficients_samples())/n
                self.drescurrent += self.current_fak * sum(self.stochastic_qs_objective.dJ_by_dcoilcurrents_samples())/n
                self.dresetabar  += sum(self.stochastic_qs_objective.dJ_by_detabar_samples())/n
                self.dresma      += sum(self.stochastic_qs_objective.dJ_by_dmagneticaxiscoefficients_samples())/n
            elif self.mode == "cvar":
                t = x[-1]
                self.res1         = self.cvar.J(t, Jsamples)
                self.res1_det     = self.res1
                self.drescoil    += self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_dcoilcoefficients_samples())
                self.drescurrent += self.current_fak * self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_dcoilcurrents_samples())
                self.dresetabar  += self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_detabar_samples())
                self.dresma      += self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_dmagneticaxiscoefficients_samples())
                self.drescvart   = self.cvar.dJ_dt(t, Jsamples)
            else:
                raise NotImplementedError

        self.Jvals_individual.append([self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res7, self.res8, self.res9, self.res_tikhonov_weight])
        self.res = sum(self.Jvals_individual[-1])
        self.perturbed_vals = [self.res - self.res1 + r for r in Jsamples]

        if self.mode in ["deterministic", "stochastic"]:
            self.dres = np.concatenate((
                self.dresetabar, self.dresma,
                self.drescurrent, self.drescoil
            ))
        elif self.mode == "cvar":
            self.dres = np.concatenate((
                self.dresetabar, self.dresma,
                self.drescurrent, self.drescoil,
                self.drescvart
            ))
        else:
            raise NotImplementedError

        if self.mode == "stochastic":
            self.dres_det = np.concatenate((
                self.dresetabar_det, self.dresma_det,
                self.drescurrent_det, self.drescoil_det
            ))


    def compute_out_of_sample(self):
        if self.stochastic_qs_objective_out_of_sample is None:
            self.stochastic_qs_objective_out_of_sample = StochasticQuasiSymmetryObjective(self.stellarator, self.sampler, self.noutsamples, self.qsf, 9999+self.seed, value_only=True)

        self.stochastic_qs_objective_out_of_sample.set_magnetic_axis(self.ma.gamma())
        Jsamples = np.array(self.stochastic_qs_objective_out_of_sample.J_samples())
        return Jsamples, Jsamples + sum(self.Jvals_individual[-1][1:])

    def callback(self, x, verbose=True):
        assert np.allclose(self.x, x)
        self.Jvals.append(self.res)
        norm = np.linalg.norm
        self.dJvals.append((
            norm(self.dres), norm(self.dresetabar), norm(self.dresma), norm(self.drescurrent), norm(self.drescoil)
        ))
        if self.ninsamples > 0:
            self.Jvals_quantiles.append((np.quantile(self.perturbed_vals, 0.1), np.mean(self.perturbed_vals), np.quantile(self.perturbed_vals, 0.9)))
        self.Jvals_no_noise.append(self.res - self.res1 + 0.5 * (self.J_BSvsQS.J_L2() + self.J_BSvsQS.J_H1()))
        self.xiterates.append(x.copy())
        if comm.rank == 0:
            self.Jvals_perturbed.append(self.perturbed_vals)

        iteration = len(self.xiterates)-1
        info("################################################################################")
        info(f"Iteration {iteration}")
        norm = np.linalg.norm
        info(f"Objective value:         {self.res:.6e}")
        info(f"Objective values:        {self.res1:.6e}, {self.res2:.6e}, {self.res3:.6e}, {self.res4:.6e}, {self.res5:.6e}, {self.res6:.6e}, {self.res7:.6e}, {self.res8:.6e}, {self.res9:.6e}, {self.res_tikhonov_weight:.6e}")
        if self.ninsamples > 0:
            info(f"VaR(.1), Mean, VaR(.9):  {np.quantile(self.perturbed_vals, 0.1):.6e}, {np.mean(self.perturbed_vals):.6e}, {np.quantile(self.perturbed_vals, 0.9):.6e}")
            v = np.asarray(self.perturbed_vals)
            cvar90 = np.mean(v[v>=np.quantile(v, 0.90)])
            cvar95 = np.mean(v[v>=np.quantile(v, 0.95)])
            info(f"CVaR(.9), CVaR(.95), Max:{cvar90:.6e}, {cvar95:.6e}, {max(self.perturbed_vals):.6e}")
        info(f"Objective gradients:     {norm(self.dresetabar):.6e}, {norm(self.dresma):.6e}, {norm(self.drescurrent):.6e}, {norm(self.drescoil):.6e}")

        max_curvature  = max(np.max(c.kappa()) for c in self.stellarator._base_coils)
        mean_curvature = np.mean([np.mean(c.kappa()) for c in self.stellarator._base_coils])
        max_torsion    = max(np.max(np.abs(c.torsion())) for c in self.stellarator._base_coils)
        mean_torsion   = np.mean([np.mean(np.abs(c.torsion())) for c in self.stellarator._base_coils])
        info(f"Curvature Max: {max_curvature:.3e}; Mean: {mean_curvature:.3e}")
        info(f"Torsion   Max: {max_torsion:.3e}; Mean: {mean_torsion:.3e}")
        # if ((iteration in list(range(6))) or iteration % self.freq_plot == 0) and self.freq_plot > 0 and comm.rank == 0:
        #     self.plot('iteration-%04i.png' % iteration)
        if iteration % self.freq_out_of_sample == 0 and self.noutsamples > 0:
            oos_vals = self.compute_out_of_sample()[1]
            if comm.rank == 0:
                self.out_of_sample_values.append(oos_vals)
            v = np.asarray(oos_vals)
            var10 = np.quantile(v, 0.10)
            var90 = np.quantile(v, 0.90)
            cvar90 = np.mean(v[v>=var90])
            var95 = np.quantile(v, 0.95)
            cvar95 = np.mean(v[v>=var95])
            info("Out of sample")
            info(f"VaR(.1), Mean, VaR(.9):  {var10:.6e}, {np.mean(oos_vals):.6e}, {var90:.6e}")
            info(f"CVaR(.9), CVaR(.95), Max:{cvar90:.6e}, {cvar95:.6e}, {max(oos_vals):.6e}")

    def plot(self, filename):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for i in range(0, len(self.stellarator.coils)):
            ax = self.stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(self.stellarator._base_coils)])
        self.ma.plot(ax=ax, show=False, closed_loop=False)
        ax.view_init(elev=90., azim=0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 1)
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        for i in range(0, len(self.stellarator.coils)):
            ax = self.stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(self.stellarator._base_coils)])
        self.ma.plot(ax=ax, show=False, closed_loop=False)
        ax.view_init(elev=0., azim=0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 1)
        plt.savefig(self.outdir + filename, dpi=300)
        plt.close()


        if "DISPLAY" in os.environ:
            try:
                import mayavi.mlab as mlab
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "\n\nPlease install mayavi first. On a mac simply do \n" +
                    "   pip3 install mayavi PyQT5\n" +
                    "On Ubuntu run \n" +
                    "   pip3 install mayavi\n" +
                    "   sudo apt install python3-pyqt4\n\n"
                )

            mlab.options.offscreen = True
            colors = [
                (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
                (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
                (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
                (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
                (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
                (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
                (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
                (0.8, 0.7254901960784313, 0.4549019607843137),
                (0.39215686274509803, 0.7098039215686275, 0.803921568627451)
            ]

            mlab.figure(bgcolor=(1, 1, 1))
            for i in range(0, len(self.stellarator.coils)):
                gamma = self.stellarator.coils[i].gamma()
                gamma = np.concatenate((gamma, [gamma[0,:]]))
                mlab.plot3d(gamma[:, 0], gamma[:, 1], gamma[:, 2], color=colors[i%len(self.stellarator._base_coils)])

            gamma = self.ma.gamma()
            theta = 2*np.pi/self.ma.nfp
            rotmat = np.asarray([
                [cos(theta), -sin(theta), 0],
                [sin(theta), cos(theta), 0],
                [0, 0, 1]]).T
            gamma0 = gamma.copy()
            for i in range(1, self.ma.nfp):
                gamma0 = gamma0 @ rotmat
                gamma = np.vstack((gamma, gamma0))
            mlab.plot3d(gamma[:, 0], gamma[:, 1], gamma[:, 2], color=colors[len(self.stellarator._base_coils)])



            mlab.view(azimuth=0, elevation=0)
            mlab.savefig(self.outdir + "mayavi_top_" + filename, magnification=4)
            mlab.view(azimuth=0, elevation=90)
            mlab.savefig(self.outdir + "mayavi_side1_" + filename, magnification=4)
            mlab.view(azimuth=90, elevation=90)
            mlab.savefig(self.outdir + "mayavi_side2_" + filename, magnification=4)
            mlab.view(azimuth=45, elevation=45)
            mlab.savefig(self.outdir + "mayavi_angled_" + filename, magnification=4)
            mlab.close()

    def save_to_matlab(self, dirname):
        dirname = os.path.join(self.outdir, dirname)
        os.makedirs(dirname, exist_ok=True)
        matlabcoils = [c.tomatlabformat() for c in self.stellarator._base_coils]
        np.savetxt(os.path.join(dirname, 'coils.txt'), np.hstack(matlabcoils))
        np.savetxt(os.path.join(dirname, 'currents.txt'), self.stellarator._base_currents)
        np.savetxt(os.path.join(dirname, 'eta_bar.txt'), [self.qsf.eta_bar])
        np.savetxt(os.path.join(dirname, 'cR.txt'), self.ma.coefficients[0])
        np.savetxt(os.path.join(dirname, 'sZ.txt'), np.concatenate(([0], self.ma.coefficients[1])))


class SimpleNearAxisQuasiSymmetryObjective():

    def __init__(self, stellarator, ma, iota_target, eta_bar=-2.25,
                 coil_length_target=None, magnetic_axis_length_target=None,
                 curvature_weight=0., torsion_weight=0., tikhonov_weight=0., arclength_weight=0., sobolev_weight=0.,
                 minimum_distance=0.04, distance_weight=0.,
                 outdir="output/"
                 ):
        self.stellarator = stellarator
        self.ma = ma
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        self.biotsavart = bs
        self.biotsavart.set_points(self.ma.gamma())
        qsf = QuasiSymmetricField(eta_bar, ma)
        self.qsf = qsf
        sigma = qsf.sigma
        iota = qsf.iota

        self.J_BSvsQS          = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
        coils = stellarator._base_coils
        self.J_coil_lengths    = [CurveLength(coil) for coil in coils]
        self.J_axis_length     = CurveLength(ma)
        if coil_length_target is not None:
            self.coil_length_targets = [coil_length_target for coil in coils]
        else:
            self.coil_length_targets = [J.J() for J in self.J_coil_lengths]
        self.magnetic_axis_length_target = magnetic_axis_length_target or self.J_axis_length.J()

        # self.J_coil_curvatures = [CurveCurvature(coil, length) for (coil, length) in zip(coils, self.coil_length_targets)]
        self.J_coil_curvatures = CoilLpReduction([CurveCurvature(coil, length, p=2, root=True) for (coil, length) in zip(coils, self.coil_length_targets)], p=2, root=True)
        # self.J_coil_torsions   = [CurveTorsion(coil, p=4) for coil in coils]
        self.J_coil_torsions   = CoilLpReduction([CurveTorsion(coil, p=2, root=True) for coil in coils], p=2, root=True)
        self.J_sobolev_weights = [SobolevTikhonov(coil, weights=[1., .1, .1, .1]) for coil in coils] + [SobolevTikhonov(ma, weights=[1., .1, .1, .1])]
        self.J_arclength_weights = [UniformArclength(coil, length) for (coil, length) in zip(coils, self.coil_length_targets)]
        self.J_distance = MinimumDistance(stellarator.coils, minimum_distance)

        self.iota_target                 = iota_target
        self.curvature_weight             = curvature_weight
        self.torsion_weight               = torsion_weight
        self.num_ma_dofs = len(ma.get_dofs())
        self.current_fak = 1./(4 * pi * 1e-7)

        self.ma_dof_idxs = (1, 1+self.num_ma_dofs)
        self.current_dof_idxs = (self.ma_dof_idxs[1], self.ma_dof_idxs[1] + len(stellarator.get_currents()))
        self.coil_dof_idxs = (self.current_dof_idxs[1], self.current_dof_idxs[1] + len(stellarator.get_dofs()))

        self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellarator.get_currents()/self.current_fak, self.stellarator.get_dofs()))
        self.x = self.x0.copy()
        self.sobolev_weight = sobolev_weight
        self.tikhonov_weight = tikhonov_weight
        self.arclength_weight = arclength_weight
        self.distance_weight = distance_weight

        self.xiterates = []
        self.Jvals_individual = []
        self.Jvals = []
        self.dJvals = []
        self.outdir = outdir

    def set_dofs(self, x):
        x_etabar = x[0]
        x_ma = x[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
        x_current = x[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
        x_coil = x[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
        self.t = x[-1]

        self.qsf.eta_bar = x_etabar
        self.ma.set_dofs(x_ma)
        self.biotsavart.set_points(self.ma.gamma())
        self.stellarator.set_currents(self.current_fak * x_current)
        self.stellarator.set_dofs(x_coil)

        self.biotsavart.clear_cached_properties()
        self.qsf.clear_cached_properties()

    def update(self, x, compute_derivative=True):
        self.x[:] = x
        J_BSvsQS          = self.J_BSvsQS
        J_coil_lengths    = self.J_coil_lengths
        J_axis_length     = self.J_axis_length
        J_coil_curvatures = self.J_coil_curvatures
        J_coil_torsions   = self.J_coil_torsions

        iota_target                 = self.iota_target
        magnetic_axis_length_target = self.magnetic_axis_length_target
        curvature_weight             = self.curvature_weight
        torsion_weight               = self.torsion_weight
        qsf = self.qsf

        self.set_dofs(x)

        self.dresetabar  = np.zeros(1)
        self.dresma      = np.zeros(self.ma_dof_idxs[1]-self.ma_dof_idxs[0])
        self.drescurrent = np.zeros(self.current_dof_idxs[1]-self.current_dof_idxs[0])
        self.drescoil    = np.zeros(self.coil_dof_idxs[1]-self.coil_dof_idxs[0])


        """ Objective values """

        self.res1        = 0.5 * J_BSvsQS.J_L2() + 0.5 * J_BSvsQS.J_H1()
        if compute_derivative:
            self.dresetabar  += 0.5 * J_BSvsQS.dJ_L2_by_detabar() + 0.5 * J_BSvsQS.dJ_H1_by_detabar()
            self.dresma      += 0.5 * J_BSvsQS.dJ_L2_by_dmagneticaxiscoefficients() + 0.5 * J_BSvsQS.dJ_H1_by_dmagneticaxiscoefficients()
            self.drescoil    += 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_L2_by_dcoilcoefficients()) \
                + 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_H1_by_dcoilcoefficients())
            self.drescurrent += 0.5 * self.current_fak * (
                self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_L2_by_dcoilcurrents()) + self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_H1_by_dcoilcurrents())
            )

        self.res2      = 0.5 * sum( (1/l)**2 * (J2.J() - l)**2 for (J2, l) in zip(J_coil_lengths, self.coil_length_targets))
        if compute_derivative:
            self.drescoil += self.stellarator.reduce_coefficient_derivatives([
                (1/l)**2 * (J_coil_lengths[i].J()-l) * J_coil_lengths[i].dJ_by_dcoefficients() for (i, l) in zip(list(range(len(J_coil_lengths))), self.coil_length_targets)])

        self.res3    = 0.5 * (1/magnetic_axis_length_target)**2 * (J_axis_length.J() - magnetic_axis_length_target)**2
        if compute_derivative:
            self.dresma += (1/magnetic_axis_length_target)**2 * (J_axis_length.J()-magnetic_axis_length_target) * J_axis_length.dJ_by_dcoefficients()

        self.res4        = 0.5 * (1/iota_target**2) * (qsf.iota-iota_target)**2
        if compute_derivative:
            self.dresetabar += (1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_detabar[:,0]
            self.dresma     += (1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_dcoeffs[:, 0]

        if curvature_weight > 0:
            # self.res5      = sum(curvature_weight * J.J() for J in J_coil_curvatures)
            self.res5      = curvature_weight  * J_coil_curvatures.J()
            if compute_derivative:
                # self.drescoil += self.curvature_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_curvatures])
                self.drescoil += self.curvature_weight * J_coil_curvatures.dJ_by_dcoefficients()
        else:
            self.res5 = 0
        if torsion_weight > 0:
            # self.res6      = sum(torsion_weight * J.J() for J in J_coil_torsions)
            self.res6 = torsion_weight * J_coil_torsions.J() 
            if compute_derivative:
                # self.drescoil += self.torsion_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_torsions])
                self.drescoil += self.torsion_weight * J_coil_torsions.dJ_by_dcoefficients() 
        else:
            self.res6 = 0

        if self.sobolev_weight > 0:
            self.res7 = sum(self.sobolev_weight * J.J() for J in self.J_sobolev_weights)
            if compute_derivative:
                self.drescoil += self.sobolev_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_sobolev_weights[:-1]])
                self.dresma += self.sobolev_weight * self.J_sobolev_weights[-1].dJ_by_dcoefficients()
        else:
            self.res7 = 0

        if self.arclength_weight > 0:
            self.res8 = sum(self.arclength_weight * J.J() for J in self.J_arclength_weights)
            if compute_derivative:
                self.drescoil += self.arclength_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_arclength_weights])
        else:
            self.res8 = 0

        if self.distance_weight > 0:
            self.res9 = self.distance_weight * self.J_distance.J()
            if compute_derivative:
                self.drescoil += self.distance_weight * self.stellarator.reduce_coefficient_derivatives(self.J_distance.dJ_by_dcoefficients())
        else:
            self.res9 = 0

        if self.tikhonov_weight > 0:
            self.res_tikhonov_weight = self.tikhonov_weight * np.sum((x-self.x0)**2)
            if compute_derivative:
                dres_tikhonov_weight = self.tikhonov_weight * 2. * (x-self.x0)
                self.dresetabar += dres_tikhonov_weight[0:1]
                self.dresma += dres_tikhonov_weight[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
                self.drescurrent += dres_tikhonov_weight[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
                self.drescoil += dres_tikhonov_weight[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
        else:
            self.res_tikhonov_weight = 0

        # self.Jvals_individual.append([])
        Jvals_individual = [self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res7, self.res8, self.res9, self.res_tikhonov_weight]
        self.res = sum(Jvals_individual)

        if compute_derivative:
            self.dres = np.concatenate((
                self.dresetabar, self.dresma,
                self.drescurrent, self.drescoil
            ))

    def clear_history(self):
        self.xiterates = []
        self.Jvals_individual = []
        self.Jvals = []
        self.dJvals = []

    def callback(self, x, verbose=True):
        self.update(x)# assert np.allclose(self.x, x)
        self.Jvals.append(self.res)
        norm = np.linalg.norm
        self.dJvals.append((
            norm(self.dres), norm(self.dresetabar), norm(self.dresma), norm(self.drescurrent), norm(self.drescoil)
        ))
        self.xiterates.append(x.copy())

        iteration = len(self.xiterates)-1
        info("################################################################################")
        info(f"Iteration {iteration}")
        norm = np.linalg.norm
        info(f"Objective value:         {self.res:.6e}")
        # info(f"Objective values:        {self.res1:.6e}, {self.res2:.6e}, {self.res3:.6e}, {self.res4:.6e}, {self.res5:.6e}, {self.res6:.6e}, {self.res7:.6e}, {self.res8:.6e}, {self.res9:.6e}, {self.res_tikhonov_weight:.6e}")
        info(f"Objective gradients:     {norm(self.dresetabar):.6e}, {norm(self.dresma):.6e}, {norm(self.drescurrent):.6e}, {norm(self.drescoil):.6e}")

        max_curvature  = max(np.max(c.kappa) for c in self.stellarator._base_coils)
        mean_curvature = np.mean([np.mean(c.kappa) for c in self.stellarator._base_coils])
        max_torsion    = max(np.max(np.abs(c.torsion)) for c in self.stellarator._base_coils)
        mean_torsion   = np.mean([np.mean(np.abs(c.torsion)) for c in self.stellarator._base_coils])
        info(f"Curvature Max: {max_curvature:.3e}; Mean: {mean_curvature:.3e}")
        info(f"Torsion   Max: {max_torsion:.3e}; Mean: {mean_torsion:.3e}")

    def plot(self, backend='plotly'):
        if backend == 'matplotlib':
            import matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            for i in range(0, len(self.stellarator.coils)):
                ax = self.stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(self.stellarator._base_coils)])
            self.ma.plot(ax=ax, show=False, closed_loop=False)
            ax.view_init(elev=90., azim=0)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1, 1)
            plt.show()
        elif backend == 'plotly':
            stellarator = self.stellarator
            coils = stellarator.coils
            ma = self.ma
            gamma = coils[0].gamma()
            N = gamma.shape[0]
            l = len(stellarator.coils)
            data = np.zeros((l*(N+1), 4))
            labels = [None for i in range(l*(N+1))]
            for i in range(l):
                data[(i*(N+1)):((i+1)*(N+1)-1),:-1] = stellarator.coils[i].gamma()
                data[((i+1)*(N+1)-1),:-1] = stellarator.coils[i].gamma()[0, :]
                data[(i*(N+1)):((i+1)*(N+1)),-1] = i
                for j in range(i*(N+1), (i+1)*(N+1)):
                    labels[j] = 'Coil %i ' % stellarator.map[i]
            N = ma.gamma().shape[0]
            ma_ = np.zeros((ma.nfp*N+1, 4))
            ma0 = ma.gamma().copy()
            theta = 2*np.pi/ma.nfp
            rotmat = np.asarray([
                [cos(theta), -sin(theta), 0],
                [sin(theta), cos(theta), 0],
                [0, 0, 1]]).T

            for i in range(ma.nfp):
                ma_[(i*N):(((i+1)*N)), :-1] = ma0
                ma0 = ma0 @ rotmat
            ma_[-1, :-1] = ma.gamma()[0,:]
            ma_[:, -1] = -1
            data = np.vstack((data, ma_))
            for i in range(ma_.shape[0]):
                labels.append('Magnetic Axis')
            import plotly.express as px
            fig = px.line_3d(x=data[:,0], y=data[:,1], z=data[:,2],
                             color=labels, line_group=data[:,3].astype(np.int))
            fig.show()
        else:
            raise NotImplementedError('backend must be either matplotlib or plotly')

def plot_stellarator(stellarator, axis=None, extra_data=None):
    coils = stellarator.coils
    gamma = coils[0].gamma()
    N = gamma.shape[0]
    l = len(stellarator.coils)
    data = np.zeros((l*(N+1), 3))
    labels = [None for i in range(l*(N+1))]
    groups = [None for i in range(l*(N+1))]
    for i in range(l):
        data[(i*(N+1)):((i+1)*(N+1)-1), :] = stellarator.coils[i].gamma()
        data[((i+1)*(N+1)-1), :] = stellarator.coils[i].gamma()[0, :]
        for j in range(i*(N+1), (i+1)*(N+1)):
            labels[j] = 'Coil %i ' % stellarator.map[i]
            groups[j] = i+1

    if axis is not None:
        N = axis.gamma().shape[0]
        ma_ = np.zeros((axis.get_nfp()*N+1, 3))
        ma0 = axis.gamma().copy()
        theta = 2*np.pi/axis.get_nfp()
        rotmat = np.asarray([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]]).T

        for i in range(axis.get_nfp()):
            ma_[(i*N):(((i+1)*N)), :] = ma0
            ma0 = ma0 @ rotmat
        ma_[-1, :] = axis.gamma()[0, :]
        data = np.vstack((data, ma_))
        for i in range(ma_.shape[0]):
            labels.append('Magnetic Axis')
            groups.append(0)

    if extra_data is not None:
        for i, extra in enumerate(extra_data):
            labels += ['Extra %i' % i ] * extra.shape[0]
            groups += [-1-i] * extra.shape[0]
            data = np.vstack((data, extra)) 
    import plotly.express as px
    fig = px.line_3d(x=data[:,0], y=data[:,1], z=data[:,2],
                     color=labels, line_group=groups)
    fig.show()
