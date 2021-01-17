"""
This script is used to compute a quadratic flux minimizing surface with 
volume 1 for the NCSX coils
"""
from pyplasmaopt import *
import numpy as np
from pyplasmaopt.grad_optimizer import GradOptimizer
from pyplasmaopt.qfm_surface import QfmSurface

Nt_ma = 6
Nt_coils = 6
ppp = 20
volume = 1.0 # Target volume
nfp = 3
mmax = 3 # maximum poloidal mode number for surface
nmax = 3 # maximum toroidal mode number for surface
ntheta = 20 # number of poloidal grid points for integration
nphi = 20 # number of toroidal grid points for integration

(coils, ma, currents) = get_ncsx_data(Nt_ma=Nt_ma, Nt_coils=Nt_coils, ppp=ppp)
stellarator = CoilCollection(coils, currents, nfp, True)
bs = BiotSavart(stellarator.coils, stellarator.currents)

qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)

objective = qfm.quadratic_flux
d_objective = qfm.d_quadratic_flux

# Initialize parameters - circular cross section torus
paramsInitR = np.zeros((qfm.mnmax))
paramsInitZ = np.zeros((qfm.mnmax))

paramsInitR[(qfm.xm==1)*(qfm.xn==0)]=0.188077
paramsInitZ[(qfm.xm==1)*(qfm.xn==0)]=-0.188077

paramsInit = np.hstack((paramsInitR[1::],paramsInitZ))

optimizer = GradOptimizer(len(paramsInit))
optimizer.add_objective(objective,d_objective,1)
optimizer.optimize(paramsInit,package='scipy',method='BFGS')