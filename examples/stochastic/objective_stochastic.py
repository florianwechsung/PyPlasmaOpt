from pyplasmaopt import *
import numpy as np
from math import pi
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD

def stochastic_get_objective():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--at-optimum", dest="at_optimum", default=False,
                        action="store_true")
    parser.add_argument("--mode", type=str, default="deterministic",
                        choices=["deterministic", "stochastic", "cvar0.5", "cvar0.9", "cvar0.95", "cvar0.99"])
    parser.add_argument("--distribution", type=str, default="gaussian",
                        choices=["gaussian", "uniform"])
    parser.add_argument("--ppp", type=int, default=20)
    parser.add_argument("--Nt_ma", type=int, default=4)
    parser.add_argument("--Nt_coils", type=int, default=4)
    parser.add_argument("--ninsamples", type=int, default=100)
    parser.add_argument("--noutsamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=3e-3)
    parser.add_argument("--length-scale", type=float, default=0.2)
    args, _ = parser.parse_known_args()

    keys = list(args.__dict__.keys())
    assert keys[0] == "output"
    if not args.__dict__[keys[0]] == "":
        outdir = "output/%s_" % args.__dict__[keys[0]]
    else:
        outdir = "output/"
    if args.__dict__[keys[1]]:
        outdir += "atopt_"
    outdir += "_".join(["%s-%s" % (keys[i], args.__dict__[keys[i]]) for i in range(2, len(keys))])
    outdir = outdir.replace(".", "p")
    outdir += "/"
    # print(f"lr {args.lr}, tau {args.tau}, c {args.c}, lam {args.lam}")
    # os.system('tail -n 1 voyager-output/' + outdir + 'out_of_sample_means.txt')
    # import sys; sys.exit()

    os.makedirs(outdir, exist_ok=True)
    set_file_logger(outdir + "log.txt")
    info("Configuration: \n%s", args.__dict__)
    
    nfp = 3
    (coils, ma, currents) = get_ncsx_data(Nt_ma=args.Nt_ma, Nt_coils=args.Nt_coils, ppp=args.ppp)
    stellarator = CoilCollection(coils, currents, nfp, True)
    eta_bar = 0.685
    iota_target = -0.395938929522566
    coil_length_target = None
    magnetic_axis_length_target = None

    # (coils, ma) = get_flat_data(ppp=args.ppp)
    # currents = [0., 0., 0.]
    # stellarator = CoilCollection(coils, currents, nfp, True)
    # iota_target = 0.45
    # coil_length_target = 2*np.pi * 0.35
    # magnetic_axis_length_target = 2*np.pi
    # eta_bar = -2.25

    # nfp = 2
    # (coils, ma, currents) = get_16_coil_data(ppp=args.ppp, at_optimum=args.at_optimum)
    # stellarator = CoilCollection(coils, currents, nfp, True)
    # coil_length_target = 2 * np.pi * 0.7
    # magnetic_axis_length_target = 2 * np.pi
    # iota_target = 0.103;
    # eta_bar = 0.998578113525166 if args.at_optimum else 1.0


    obj = NearAxisQuasiSymmetryObjective(
        stellarator, ma, iota_target, eta_bar=eta_bar,
        coil_length_target=coil_length_target, magnetic_axis_length_target=magnetic_axis_length_target,
        # curvature_weight=args.curvature, torsion_weight=args.torsion,
        # tikhonov_weight=args.tikhonov, arclength_weight=args.arclength, sobolev_weight=args.sobolev,
        # minimum_distance=args.min_dist, distance_weight=args.dist_weight,
        ninsamples=args.ninsamples, noutsamples=args.noutsamples,
        sigma_perturb=args.sigma, length_scale_perturb=args.length_scale,
        mode=args.mode, outdir=outdir, seed=args.seed,
        distribution=args.distribution)
    return obj, args
