import numpy as np
from .curve import GaussianPerturbedCurve
from .objective import BiotSavartQuasiSymmetricFieldDifference
from .biotsavart import BiotSavart
from .cvar import CVaR
from property_manager3 import cached_property, PropertyManager
from mpi4py import MPI
comm = MPI.COMM_WORLD
from randomgen import Generator, PCG64


class StochasticQuasiSymmetryObjective(PropertyManager):

    def __init__(self, stellarator, sampler, nsamples, qsf, seed):
        self.stellarator = stellarator
        self.nsamples = nsamples
        size = comm.size
        idxs = [i*nsamples//size for i in range(size+1)]
        assert idxs[0] == 0
        assert idxs[-1] == nsamples
        first = idxs[comm.rank]
        last = idxs[comm.rank+1]
        assert last >= first
        self.J_BSvsQS_perturbed = []

        for i in range(first, last):
            rg = np.random.Generator(PCG64(seed, i, mode="sequence"))
            perturbed_coils = [
                GaussianPerturbedCurve(coil, sampler, randomgen=rg) for coil in stellarator.coils]
            perturbed_bs    = BiotSavart(perturbed_coils, stellarator.currents)
            self.J_BSvsQS_perturbed.append(BiotSavartQuasiSymmetricFieldDifference(qsf, perturbed_bs))

    def resample(self):
        for J in self.J_BSvsQS_perturbed:
            for c in J.biotsavart.coils:
                c.resample()

    def set_magnetic_axis(self, gamma):
        for J in self.J_BSvsQS_perturbed:
            J.biotsavart.clear_cached_properties()
            J.biotsavart.set_points(gamma)

    def J_samples(self):
        local_vals = [0.5 * (J.J_L2() + J.J_H1()) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        return all_vals

    def dJ_by_dcoilcoefficients_samples(self, t=None):
        local_vals = [0.5 * 
                      (
                          self.stellarator.reduce_coefficient_derivatives(J.dJ_L2_by_dcoilcoefficients())
                          + self.stellarator.reduce_coefficient_derivatives(J.dJ_H1_by_dcoilcoefficients())
                      ) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        return all_vals

    def dJ_by_dcoilcurrents_samples(self, t=None):
        local_vals = [0.5 * 
                      (
                          self.stellarator.reduce_current_derivatives(J.dJ_L2_by_dcoilcurrents())
                          + self.stellarator.reduce_current_derivatives(J.dJ_H1_by_dcoilcurrents())
                      ) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        return all_vals

    def dJ_by_detabar_samples(self, t=None):
        local_vals = [0.5 * (J.dJ_L2_by_detabar() + J.dJ_H1_by_detabar()) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        return all_vals

    def dJ_by_dmagneticaxiscoefficients_samples(self, t=None):
        local_vals = [0.5 * (J.dJ_L2_by_dmagneticaxiscoefficients() + J.dJ_H1_by_dmagneticaxiscoefficients()) for J in self.J_BSvsQS_perturbed]
        all_vals = [i for o in comm.allgather(local_vals) for i in o]
        return all_vals
