import numpy as np
from .curve import GaussianPerturbedCurve
from .objective import BiotSavartQuasiSymmetricFieldDifference
from .biotsavart import BiotSavart
from .cvar import CVaR
from property_manager3 import cached_property, PropertyManager
from mpi4py import MPI
comm = MPI.COMM_WORLD


class StochasticQuasiSymmetryObjective(PropertyManager):

    def __init__(self, stellarator, sampler, nsamples, qsf, mode):
        self.stellarator = stellarator

        self.J_BSvsQS_perturbed = []
        for i in range(nsamples):
            perturbed_coils = [GaussianPerturbedCurve(coil, sampler) for coil in stellarator.coils]
            perturbed_bs    = BiotSavart(perturbed_coils, stellarator.currents)
            self.J_BSvsQS_perturbed.append(BiotSavartQuasiSymmetricFieldDifference(qsf, perturbed_bs))

        size = comm.size
        idxs = [i*nsamples//size for i in range(size+1)]
        assert idxs[0] == 0
        assert idxs[-1] == nsamples
        first = idxs[comm.rank]
        last = idxs[comm.rank+1]
        assert last > first
        self.J_BSvsQS_perturbed = self.J_BSvsQS_perturbed[first:last]
        self.mode = mode

    def set_magnetic_axis(self, gamma):
        self.clear_cached_properties()
        for J in self.J_BSvsQS_perturbed:
            J.biotsavart.clear_cached_properties()
            J.biotsavart.set_points(gamma)

    @cached_property
    def all_Js(self):
        local_vals = [0.5 * (J.J_L2() + J.J_H1()) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        return all_vals


    def J(self, t=None):
        all_vals = self.all_Js
        if self.mode == "stochastic":
            return np.mean(all_vals)
        elif isinstance(self.mode, CVaR):
            return self.mode.J(t, all_vals)
        else:
            raise NotImplementedError

    def dJ_by_dcoilcoefficients(self, t=None):
        local_vals = [0.5 * 
                      (
                          self.stellarator.reduce_coefficient_derivatives(J.dJ_L2_by_dcoilcoefficients())
                          + self.stellarator.reduce_coefficient_derivatives(J.dJ_H1_by_dcoilcoefficients())
                      ) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        if self.mode == "stochastic":
            return sum(all_vals)/len(all_vals)
        elif isinstance(self.mode, CVaR):
            return self.mode.dJ_dx(t, self.all_Js, all_vals)
        else:
            raise NotImplementedError

    def dJ_by_dcoilcurrents(self, t=None):
        local_vals = [0.5 * 
                      (
                          self.stellarator.reduce_current_derivatives(J.dJ_L2_by_dcoilcurrents())
                          + self.stellarator.reduce_current_derivatives(J.dJ_H1_by_dcoilcurrents())
                      ) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        if self.mode == "stochastic":
            return sum(all_vals)/len(all_vals)
        elif isinstance(self.mode, CVaR):
            return self.mode.dJ_dx(t, self.all_Js, all_vals)
        else:
            raise NotImplementedError

    def dJ_by_detabar(self, t=None):
        local_vals = [0.5 * (J.dJ_L2_by_detabar() + J.dJ_H1_by_detabar()) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        if self.mode == "stochastic":
            return sum(all_vals)/len(all_vals)
        elif isinstance(self.mode, CVaR):
            return self.mode.dJ_dx(t, self.all_Js, all_vals)
        else:
            raise NotImplementedError

    def dJ_by_dmagneticaxiscoefficients(self, t=None):
        local_vals = [0.5 * (J.dJ_L2_by_dmagneticaxiscoefficients() + J.dJ_H1_by_dmagneticaxiscoefficients()) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        if self.mode == "stochastic":
            return sum(all_vals)/len(all_vals)
        elif isinstance(self.mode, CVaR):
            return self.mode.dJ_dx(t, self.all_Js, all_vals)
        else:
            raise NotImplementedError

    def dJ_by_dt(self, t=None):
        if isinstance(self.mode, CVaR):
            return self.mode.dJ_dt(t, self.all_Js)
        else:
            raise NotImplementedError

    def find_optimal_t(self, tinit=None):
        if not isinstance(self.mode, CVaR):
            raise NotImplementedError
        if tinit is None:
            tinit = np.asarray([0.])
        else:
            tinit = np.asarray([tinit])
        from scipy.optimize import minimize
        def J(t):
            return self.J(t=t), self.dJ_by_dt(t=t)
        res = minimize(J, tinit, jac=True, method='BFGS', tol=1e-10, options={'maxiter': 100})
        t = res.x[0]
        return t
