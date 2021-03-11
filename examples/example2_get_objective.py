from pyplasmaopt import *
import numpy as np
from math import pi
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD

def example2_get_objective():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--at-optimum", dest="at_optimum", default=False,
                        action="store_true")
    parser.add_argument("--mode", type=str, default="deterministic",
                        choices=["deterministic", "stochastic", "cvar0.5", "cvar0.9", "cvar0.95"])
    parser.add_argument("--distribution", type=str, default="gaussian",
                        choices=["gaussian", "uniform"])
    parser.add_argument("--sigma", type=float, default=3e-3)
    parser.add_argument("--length-scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ppp", type=int, default=20)
    parser.add_argument("--ninsamples", type=int, default=100)
    parser.add_argument("--noutsamples", type=int, default=100)
    parser.add_argument("--curvature", type=float, default=0.)
    parser.add_argument("--torsion", type=float, default=0.)
    parser.add_argument("--tikhonov", type=float, default=0.)
    parser.add_argument("--sobolev", type=float, default=0.)
    parser.add_argument("--arclength", type=float, default=0.)
    parser.add_argument("--min-dist", type=float, default=0.04)
    parser.add_argument("--dist-weight", type=float, default=0.)
    parser.add_argument("--optimizer", type=str, default="bfgs", choices=["bfgs", "lbfgs", "sgd", 'l-bfgs-b', 'newton-cg'])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=100)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=1e-5)
    args, _ = parser.parse_known_args()

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
    # print(f"lr {args.lr}, tau {args.tau}, c {args.c}, lam {args.lam}")
    # os.system('tail -n 1 voyager-output/' + outdir + 'out_of_sample_means.txt')
    # import sys; sys.exit()

    os.makedirs(outdir, exist_ok=True)
    set_file_logger(outdir + "log.txt")
    info("Configuration: \n%s", args.__dict__)
    
    nfp = 2
    order = 5
    (coils, currents, ma, eta_bar) = get_24_coil_data(Nt_coils=order, Nt_ma=1, nfp=nfp, ppp=args.ppp, at_optimum=args.at_optimum)
    stellarator = CoilCollection(coils, currents, nfp, True)

    iota_target = 0.103
    coil_length_target = 4.398229715025710
    magnetic_axis_length_target = 6.356206812106860
    eta_bar = -2.25
    obj = NearAxisQuasiSymmetryObjective(
        stellarator, ma, iota_target, eta_bar=eta_bar,
        coil_length_target=coil_length_target, magnetic_axis_length_target=magnetic_axis_length_target,
        curvature_weight=args.curvature, torsion_weight=args.torsion,
        tikhonov_weight=args.tikhonov, arclength_weight=args.arclength, sobolev_weight=args.sobolev,
        minimum_distance=args.min_dist, distance_weight=args.dist_weight,
        ninsamples=args.ninsamples, noutsamples=args.noutsamples, sigma_perturb=args.sigma,
        length_scale_perturb=args.length_scale, mode=args.mode, outdir=outdir, seed=args.seed,
        distribution=args.distribution, freq_plot=50)
    return obj, args

