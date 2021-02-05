from pyplasmaopt import *
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--case", type=str, default="ncsx-orig")
parser.add_argument("--energy", type=float, default=1e3)
parser.add_argument("--tmax", type=float, default=1e-2)
parser.add_argument("--nparticles", type=int, default=20)
args, _ = parser.parse_known_args()

tmax = args.tmax
nparticles = args.nparticles
energy = args.energy
def get_ncsx_data(Nt_coils=25, ppp=10, case='orig'):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if case == 'orig':
        coil_data = np.loadtxt(os.path.join(dir_path, "ncsxdata", "NCSX_coil_coeffs.dat"), delimiter=',')
        currents = np.loadtxt(os.path.join(dir_path, "ncsxdata", "NCSX_I.dat"), delimiter=',')
    elif case == 'axis':
        coil_data = np.loadtxt(os.path.join(dir_path, "ncsxdata", "NCSX_axis_coil_coeffs.dat"), delimiter=',')
        currents = np.loadtxt(os.path.join(dir_path, "ncsxdata", "NCSX_axis_I.dat"), delimiter=',')
    elif case == 'surf':
        coil_data = np.loadtxt(os.path.join(dir_path, "ncsxdata", "NCSX_surf_coil_coeffs.dat"), delimiter=',')
        currents = np.loadtxt(os.path.join(dir_path, "ncsxdata", "NCSX_surf_I.dat"), delimiter=',')
    else:
        raise NotImplementedError

    nfp = 3
    num_coils = 3
    coils = [FourierCurve(Nt_coils*ppp, Nt_coils) for i in range(num_coils)]
    for ic in range(num_coils):
        dofs = coils[ic].dofs
        dofs[0][0] = coil_data[0, 6*ic + 1]
        dofs[1][0] = coil_data[0, 6*ic + 3]
        dofs[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, Nt_coils):
            dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].set_dofs(np.concatenate(dofs))

    return (coils, currents)




def run_tracing(bs, ma=None, nparticles=401, tmax=1e-2, seed=1, outdir="", filename=""):
    if ma is None:
        rguess = 1.5
        axis = find_magnetic_axis(bs, 100, rguess, output='cartesian')
    else:
        axis = ma.gamma()

    # plot_stellarator(coil_collection, extra_data=[axis])

    mode = 'gyro'
    res, res_t, us = trace_particles_on_axis(axis, bs, nparticles, mode=mode, tmax=tmax, seed=seed, Ekinev=energy, umin=-1, umax=1)
    # plot_stellarator(coil_collection, extra_data=[axis] + res)
    return res, res_t, us

titles = {'ncsx-orig': 'orig', 'andrew-axis':'axis', 'andrew-surfaces':'surf'}
if args.case in titles:
    outdir = 'confinement'
    os.makedirs(outdir, exist_ok=True)
    title = titles[args.case]
    info(f"title = {title}")
    outdir = ""
    coils, currents = get_ncsx_data(Nt_coils=5, ppp=10, case=title)
    stellarator = CoilCollection(coils, currents, nfp=3, stellarator_symmetry=True)
    bs0 = BiotSavart(stellarator.coils, stellarator.currents)

    particleseed = comm.rank
    res_t_list = []
    us_list = []
    labels = []
    rg = np.random.Generator(PCG64(0, 9999, mode="sequence"))
    length_scale_perturb = 0.2
    sigma_perturb = 0.01
    sampler = GaussianSampler(coils[0].quadpoints, length_scale=length_scale_perturb, sigma=sigma_perturb)
    for i in [None] + list(range(5)):
        if i == None:
            bs = bs0
        else:
            perturbed_coils = [RandomlyPerturbedCurve(coil, sampler, randomgen=rg) for coil in stellarator.coils]
            bs    = BiotSavart(perturbed_coils, stellarator.currents)
        filename = f"tracing_{title}_{energy:0.f}eV_coilseed_{i}"
        local_res, local_res_t, local_us = run_tracing(bs, ma=None, nparticles=nparticle, tmax=tmax, seed=particleseed, outdir=outdir, filename=filename)
        res = np.asarray([i for o in comm.allgather(local_res) for i in o])
        res_t = np.asarray([i for o in comm.allgather(local_res_t) for i in o])
        us = np.asarray([i for o in comm.allgather(local_us) for i in o])
        info(f"res.shape={res.shape}, res_t.shape={res_t.shape}, us.shape={us.shape}")
        res_t_list.append(res_t)
        us_list.append(us)
        labels.append(filename)
        if comm.rank == 0:
            np.save(f"{outdir}/{filename}.npy", np.asarray([res_t, us]))
    import sys; sys.exit()

outdir = args.case
# outdir = "output/Hk_atopt_mode-deterministic_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-1_noutsamples-8_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-0p0_ig-0_ip-l2_optim-scipy/"
# outdir = "output/Hk_atopt_mode-stochastic_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-64_noutsamples-64_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-0p0_ig-0_ip-l2_optim-scipy/"
# outdir = "output/Hk_atopt_mode-stochastic_distribution-uniform_ppp-10_Nt_ma-4_Nt_coils-6_ninsamples-1024_noutsamples-1024_seed-1_sigma-0p01_length_scale-0p2_tikhonov-0p0_curvature-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-0p0_ig-0_ip-l2_optim-scipy/"

import sys
sys.argv = sys.argv[:1] + [str(s) for s in np.loadtxt(outdir + 'argv.txt', dtype=np.dtype('<U26'))] 
from objective_stochastic import stochastic_get_objective
obj, args = stochastic_get_objective()
x = np.load(outdir + "xiterates.npy")[-1, :]
obj.update(x)
obj.compute_out_of_sample()
info_all_sync(f"f(x) = {obj.res}")


particleseed = comm.rank
res_t_list = []
us_list = []
labels = []
rg = np.random.Generator(PCG64(0, 9999, mode="sequence"))
sampler = obj.sampler
for i in [None] + list(range(5)):
    if i is None:
        J = obj.J_BSvsQS
        info_all_sync(f'Quasi symmetry: {J.J_H1()+J.J_L2()}')
        bs = J.biotsavart
    else:
        perturbed_coils = [RandomlyPerturbedCurve(coil, sampler, randomgen=rg) for coil in obj.stellarator.coils]
        bs    = BiotSavart(perturbed_coils, obj.stellarator.currents)
        bs.set_points(obj.qsf.magnetic_axis.gamma())
        J = BiotSavartQuasiSymmetricFieldDifference(obj.qsf, bs, value_only=True)
        info_all_sync(f'Quasi symmetry: {J.J_H1()+J.J_L2()}')
        bs = J.biotsavart
    filename = f"tracing_{energy:0.f}eV_coilseed_{i}"
    local_res, local_res_t, local_us = run_tracing(bs, ma=None, nparticles=nparticles, tmax=tmax, seed=particleseed, outdir=outdir, filename=filename)
    res = np.asarray([i for o in comm.allgather(local_res) for i in o])
    res_t = np.asarray([i for o in comm.allgather(local_res_t) for i in o])
    us = np.asarray([i for o in comm.allgather(local_us) for i in o])
    info(f"res.shape={res.shape}, res_t.shape={res_t.shape}, us.shape={us.shape}")
    res_t_list.append(res_t)
    us_list.append(us)
    labels.append(filename)
    if comm.rank == 0:
        np.save(f"{outdir}{filename}.npy", np.asarray([res_t, us]))
