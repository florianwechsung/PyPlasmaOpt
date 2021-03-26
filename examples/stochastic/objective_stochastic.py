from pyplasmaopt import *
from simsopt.geo.curverzfourier import CurveRZFourier
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
    parser.add_argument("--config", type=str, default="ncsx",
                        choices=["ncsx", "matt24"])
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
    #parser.add_argument("--length-scale", type=float, default=0.2)
    
    parser.add_argument("--tikhonov", type=float, default=0.)
    parser.add_argument("--curvature", type=float, default=0.)
    parser.add_argument("--torsion", type=float, default=0.)
    parser.add_argument("--sobolev", type=float, default=0.)
    parser.add_argument("--carclen", type=float, default=0.)
    parser.add_argument("--clen", type=float, default=1.)
    parser.add_argument("--distw", type=float, default=0.)

    parser.add_argument("--ig", type=int, default=0)
    parser.add_argument("--ip", type=str, choices=["l2", "Hk"], default="l2")
    parser.add_argument("--optim", type=str, choices=["pylbfgs", "scipy"], default="scipy")
    
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
    

    Nt_ma = args.Nt_ma
    if args.config == "ncsx":
        nfp = 3
        (coils, ma, currents) = get_ncsx_data(Nt_ma=Nt_ma, Nt_coils=args.Nt_coils, ppp=args.ppp)
        eta_bar = 0.685
        iota_target = -0.395938929522566
        magnetic_axis_length_target = None
    elif args.config == "matt24":
        nfp = 2
        (coils, currents, ma, eta_bar) = get_24_coil_data(Nt_coils=args.Nt_coils, Nt_ma=args.Nt_ma, nfp=nfp, ppp=args.ppp, at_optimum=args.at_optimum)
        iota_target = 0.103
        coil_length_target = 4.398229715025710
        magnetic_axis_length_target = 6.356206812106860
    else:
        raise NotImplementedError




    stellarator = CoilCollection(coils, currents, nfp, True)
    # plot_stellarator(stellarator, axis=ma)


    # nfp = 2
    # (coils, ma, currents) = get_16_coil_data(ppp=args.ppp, at_optimum=args.at_optimum)
    # stellarator = CoilCollection(coils, currents, nfp, True)
    # coil_length_target = 2 * np.pi * 0.7
    # magnetic_axis_length_target = 2 * np.pi
    # iota_target = 0.103;
    # eta_bar = 0.998578113525166 if args.at_optimum else 1.0


    # to make sure the number of quad points is odd TODO: figure out why that's
    # important at some point... for some reason the ricatti solver fails on
    # even number of quadrature points.
    shift = 1-nfp%2 

    mafull = CurveRZFourier(nfp*len(ma.quadpoints)+shift, nfp*Nt_ma, 1, True)
    for i in range(Nt_ma):
        mafull.rc[nfp*i] = ma.rc[i]
        mafull.zs[nfp*i+nfp-1] = ma.zs[i]
    mafull.rc[nfp*Nt_ma] = ma.rc[Nt_ma]
    ma = mafull

    coil_length_target = [CurveLength(coil).J() for coil in coils]

    obj = NearAxisQuasiSymmetryObjective(
        stellarator, ma, iota_target, eta_bar=eta_bar,
        coil_length_target=coil_length_target, magnetic_axis_length_target=magnetic_axis_length_target,
        coil_length_weight=args.clen, axis_length_weight=1.,
        torsion_weight=args.torsion,
        curvature_weight=args.curvature,
        tikhonov_weight=args.tikhonov,
        arclength_weight=args.carclen,
        minimum_distance=0.2, distance_weight=args.distw,
        sobolev_weight=args.sobolev,
        ninsamples=args.ninsamples, noutsamples=args.noutsamples,
        sigma_perturb=args.sigma, length_scale_perturb=0.2,
        mode=args.mode, outdir=outdir, seed=args.seed,
        distribution=args.distribution, innerproduct=args.ip)

    return obj, args
