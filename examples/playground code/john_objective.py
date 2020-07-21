from pyplasmaopt import *
import numpy as np
from math import pi
import argparse
#Not sure if I need these: have to do with parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD

def get_objective():

    #generate output directory
    outdir = "john_test_output"
    os.makedirs(outdir, exist_ok=True)

    #PyPlasmaOpt function that logs stuff?
    set_file_logger(outdir + "log.txt")
    info("Lets get this puppy started. Normally where the configuration would be specified.")

    #number of field points
    nfp = 2

    #ppp: points per period?
    ppp = 10
    
    #another PyPlasmaOpt function retreiving some initial data provided by Dr. Landreman
    (coils, ma) = get_matt_data(nfp=nfp, ppp=ppp, at_optimum=False)

    #Taken directly from problem2_objective.py, not sure why this is necessary
    currents = [0 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
    eta_bar = -2.25

    print (currents)

get_objective()