from pyplasmaopt import *
import numpy as np
from math import pi
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD

def get_objective():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--at-optimum", dest="at_optimum", default=False,
                        action="store_true")
    parser.add_argument("--mode", type=str, default="deterministic",
                        choices=["deterministic", "stochastic", "cvar0.5", "cvar0.9", "cvar0.95"])
    parser.add_argument("--sigma", type=float, default=3e-3)
    parser.add_argument("--length-scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ppp", type=int, default=20)
    parser.add_argument("--ninsamples", type=int, default=100)
    parser.add_argument("--noutsamples", type=int, default=100)
    parser.add_argument("--curvature-pen", type=float, default=0.)
    parser.add_argument("--torsion-pen", type=float, default=0.)
    parser.add_argument("--tikhonov", type=float, default=0.)
    parser.add_argument("--sobolev", type=float, default=0.)
    parser.add_argument("--arclength", type=float, default=0.)
    parser.add_argument("--min-dist", type=float, default=0.04)
    parser.add_argument("--dist-weight", type=float, default=0.)
    parser.add_argument("--optimizer", type=str, default="bfgs", choices=["bfgs", "lbfgs", "sgd"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=100)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=1e-5)
    args, _ = parser.parse_known_args()

    nfp = 2
    (coils, ma) = get_matt_data(nfp=nfp, ppp=args.ppp, at_optimum=args.at_optimum)
    if args.at_optimum:
        currents = [1e5 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
        eta_bar = -2.105800979374183
    else:
        currents = [0 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
        eta_bar = -2.25
    stellarator = CoilCollection(coils, currents, nfp, True)
    keys = list(args.__dict__.keys())
    assert keys[0] == "output"
    if not args.__dict__[keys[0]] == "":
        outdir = "output-%s" % args.__dict__[keys[0]]
    else:
        outdir = "output"
    if args.__dict__[keys[1]]:
        outdir += "_atopt"
    for i in range(2, len(keys)):
        k = keys[i]
        outdir += "_%s-%s" % (k, args.__dict__[k])
    outdir = outdir.replace(".", "p")
    outdir += "/"
    os.makedirs(outdir, exist_ok=True)
    set_file_logger(outdir + "log.txt")
    info("Configuration: \n%s", args.__dict__)
    obj = Problem2_Objective(
        stellarator, ma, curvature_scale=args.curvature_pen, torsion_scale=args.torsion_pen,
        tikhonov=args.tikhonov, arclength=args.arclength, sobolev=args.sobolev,
        minimum_distance=args.min_dist, distance_weight=args.dist_weight,
        eta_bar=eta_bar, ninsamples=args.ninsamples, noutsamples=args.noutsamples, sigma_perturb=0.003,#args.sigma,
        length_scale_perturb=args.length_scale, mode=args.mode, outdir=outdir, seed=args.seed)
    return obj, args

class Problem2_Objective():

    def __init__(self, stellarator, ma, 
                 iota_target=0.103, coil_length_target=4.398229715025710, magnetic_axis_length_target=6.356206812106860,
                 eta_bar=-2.25,
                 curvature_scale=1e-6, torsion_scale=1e-4, tikhonov=0., arclength=0., sobolev=0.,
                 minimum_distance=0.04, distance_weight=1.,
                 ninsamples=0, noutsamples=0, sigma_perturb=1e-4, length_scale_perturb=0.2, mode="deterministic",
                 outdir="output/", seed=1
                 ):
        self.stellarator = stellarator
        self.seed = seed
        self.ma = ma
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        self.biotsavart = bs
        self.biotsavart.set_points(self.ma.gamma)
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
        self.J_coil_curvatures = [CurveCurvature(coil, coil_length_target) for coil in coils]
        self.J_coil_torsions   = [CurveTorsion(coil, p=4) for coil in coils]
        self.J_sobolevs = [SobolevTikhonov(coil, weights=[1., .1, .1, .1]) for coil in coils] + [SobolevTikhonov(ma, weights=[1., .1, .1, .1])]
        self.J_arclengths = [UniformArclength(coil, coil_length_target) for coil in coils]
        self.J_distance = MinimumDistance(stellarator.coils, minimum_distance)

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
        if mode in ["deterministic", "stochastic"]:
            self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellarator.get_currents()/self.current_fak, self.stellarator.get_dofs()))
        elif mode[0:4] == "cvar":
            self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellarator.get_currents()/self.current_fak, self.stellarator.get_dofs(), [0.]))
        else:
            raise NotImplementedError
        self.x = self.x0.copy()
        self.sobolev = sobolev
        self.tikhonov = tikhonov
        self.arclength = arclength
        self.distance_weight = distance_weight

        sampler = GaussianSampler(coils[0].points, length_scale=length_scale_perturb, sigma=sigma_perturb)
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

        self.stochastic_qs_objective.set_magnetic_axis(self.ma.gamma)

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
        self.biotsavart.set_points(self.ma.gamma)
        self.stellarator.set_currents(self.current_fak * x_current)
        self.stellarator.set_dofs(x_coil)

        self.biotsavart.clear_cached_properties()
        self.qsf.clear_cached_properties()

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

        self.set_dofs(x)

        self.dresetabar  = np.zeros(1)
        self.dresma      = np.zeros(self.ma_dof_idxs[1]-self.ma_dof_idxs[0])
        self.drescurrent = np.zeros(self.current_dof_idxs[1]-self.current_dof_idxs[0])
        self.drescoil    = np.zeros(self.coil_dof_idxs[1]-self.coil_dof_idxs[0])


        """ Objective values """
        self.stochastic_qs_objective.set_magnetic_axis(self.ma.gamma)

        Jsamples = self.stochastic_qs_objective.J_samples()
        assert len(Jsamples) == self.ninsamples
        self.QSvsBS_perturbed.append(Jsamples)
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
            n = self.ninsamples
            self.res1         = sum(Jsamples)/n
            self.drescoil    += sum(self.stochastic_qs_objective.dJ_by_dcoilcoefficients_samples())/n
            self.drescurrent += self.current_fak * sum(self.stochastic_qs_objective.dJ_by_dcoilcurrents_samples())/n
            self.dresetabar  += sum(self.stochastic_qs_objective.dJ_by_detabar_samples())/n
            self.dresma      += sum(self.stochastic_qs_objective.dJ_by_dmagneticaxiscoefficients_samples())/n
        elif self.mode == "cvar":
            t = x[-1]
            self.res1         = self.cvar.J(t, Jsamples)
            self.drescoil    += self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_dcoilcoefficients_samples())
            self.drescurrent += self.current_fak * self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_dcoilcurrents_samples())
            self.dresetabar  += self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_detabar_samples())
            self.dresma      += self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_dmagneticaxiscoefficients_samples())
            self.drescvart   = self.cvar.dJ_dt(t, Jsamples)
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

        if curvature_scale > 1e-15:
            self.res5      = sum(curvature_scale * J.J() for J in J_coil_curvatures)
            self.drescoil += self.curvature_scale * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_curvatures])
        else:
            self.res5 = 0
        if torsion_scale > 1e-15:
            self.res6      = sum(torsion_scale * J.J() for J in J_coil_torsions)
            self.drescoil += self.torsion_scale * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_torsions])
        else:
            self.res6 = 0

        if self.sobolev > 1e-15:
            self.res7 = sum(self.sobolev * J.J() for J in self.J_sobolevs)
            self.drescoil += self.sobolev * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_sobolevs[:-1]])
            self.dresma += self.sobolev * self.J_sobolevs[-1].dJ_by_dcoefficients()
        else:
            self.res7 = 0

        if self.arclength > 1e-15:
            self.res8 = sum(self.arclength * J.J() for J in self.J_arclengths)
            self.drescoil += self.arclength * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_arclengths])
        else:
            self.res8 = 0

        if self.distance_weight > 1e-15:
            self.res9 = self.distance_weight * self.J_distance.J()
            self.drescoil += self.distance_weight * self.stellarator.reduce_coefficient_derivatives(self.J_distance.dJ_by_dcoefficients())
        else:
            self.res9 = 0

        if self.tikhonov > 1e-15:
            self.res_tikhonov = self.tikhonov * np.sum((x-self.x0)**2)
            dres_tikhonov = self.tikhonov * 2. * (x-self.x0)
            self.dresetabar += dres_tikhonov[0:1]
            self.dresma += dres_tikhonov[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
            self.drescurrent += dres_tikhonov[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
            self.drescoil += dres_tikhonov[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
        else:
            self.res_tikhonov = 0

        self.Jvals_individual.append([self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res7, self.res8, self.res9, self.res_tikhonov])
        self.res = sum(self.Jvals_individual[-1])
        self.perturbed_vals = [self.res - self.res1 + r for r in self.QSvsBS_perturbed[-1]]

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

    def compute_out_of_sample(self):
        if self.stochastic_qs_objective_out_of_sample is None:
            self.stochastic_qs_objective_out_of_sample = StochasticQuasiSymmetryObjective(self.stellarator, self.sampler, self.noutsamples, self.qsf, 9999+self.seed)

        self.stochastic_qs_objective_out_of_sample.set_magnetic_axis(self.ma.gamma)
        Jsamples = np.array(self.stochastic_qs_objective_out_of_sample.J_samples())
        return Jsamples, Jsamples + sum(self.Jvals_individual[-1][1:])

    def callback(self, x, verbose=True):
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

        iteration = len(self.xiterates)-1
        info("################################################################################")
        info(f"Iteration {iteration}")
        norm = np.linalg.norm
        info(f"Objective value:         {self.res:.6e}")
        info(f"Objective values:        {self.res1:.6e}, {self.res2:.6e}, {self.res3:.6e}, {self.res4:.6e}, {self.res5:.6e}, {self.res6:.6e}, {self.res7:.6e}, {self.res8:.6e}, {self.res9:.6e}, {self.res_tikhonov:.6e}")
        info(f"VaR(.1), Mean, VaR(.9):  {np.quantile(self.perturbed_vals, 0.1):.6e}, {np.mean(self.perturbed_vals):.6e}, {np.quantile(self.perturbed_vals, 0.9):.6e}")
        cvar90 = np.mean(list(v for v in self.perturbed_vals if v >= np.quantile(self.perturbed_vals, 0.9)))
        cvar95 = np.mean(list(v for v in self.perturbed_vals if v >= np.quantile(self.perturbed_vals, 0.95)))
        info(f"CVaR(.9), CVaR(.95), Max:{cvar90:.6e}, {cvar95:.6e}, {max(self.perturbed_vals):.6e}")
        info(f"Objective gradients:     {norm(self.dresetabar):.6e}, {norm(self.dresma):.6e}, {norm(self.drescurrent):.6e}, {norm(self.drescoil):.6e}")

        max_curvature  = max(np.max(c.kappa) for c in self.stellarator._base_coils)
        mean_curvature = np.mean([np.mean(c.kappa) for c in self.stellarator._base_coils])
        max_torsion    = max(np.max(np.abs(c.torsion)) for c in self.stellarator._base_coils)
        mean_torsion   = np.mean([np.mean(np.abs(c.torsion)) for c in self.stellarator._base_coils])
        info(f"Curvature Max: {max_curvature:.3e}; Mean: {mean_curvature:.3e}")
        info(f"Torsion   Max: {max_torsion:.3e}; Mean: {mean_torsion:.3e}")
        if iteration % 25 == 0:
            oos_vals = self.compute_out_of_sample()[1]
            self.out_of_sample_values.append(oos_vals)
            info("Out of sample")
            info(f"VaR(.1), Mean, VaR(.9):  {np.quantile(oos_vals, 0.1):.6e}, {np.mean(oos_vals):.6e}, {np.quantile(oos_vals, 0.9):.6e}")
            info(f"CVaR(.9), CVaR(.95), Max:{np.mean(list(v for v in oos_vals if v >= np.quantile(oos_vals, 0.9))):.6e}, {np.mean(list(v for v in oos_vals if v >= np.quantile(oos_vals, 0.95))):.6e}, {max(oos_vals):.6e}")
            if comm.rank == 0:
                self.plot('iteration-%04i.png' % iteration)

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

            for i in range(0, len(self.stellarator.coils)):
                gamma = self.stellarator.coils[i].gamma
                gamma = np.concatenate((gamma, [gamma[0,:]]))
                mlab.plot3d(gamma[:, 0], gamma[:, 1], gamma[:, 2], color=colors[i%len(self.stellarator._base_coils)])
            mlab.view(azimuth=0, elevation=0)
            mlab.savefig(self.outdir + "mayavi_top_" + filename, magnification=4)
            mlab.view(azimuth=0, elevation=90)
            mlab.savefig(self.outdir + "mayavi_side1_" + filename, magnification=4)
            mlab.view(azimuth=90, elevation=90)
            mlab.savefig(self.outdir + "mayavi_side2_" + filename, magnification=4)
            mlab.close()
