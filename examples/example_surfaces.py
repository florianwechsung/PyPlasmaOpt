from pyplasmaopt import *
from simsgeo import JaxStelleratorSymmetricCylindricalFourierCurve, StelleratorSymmetricCylindricalFourierCurve, \
FourierCurve, JaxFourierCurve, RotatedCurve, JaxCartesianSurface, stelleratorsymmetriccylindricalfouriercurve_pure, \
JaxCartesianMagneticSurface
import jax.numpy as jnp
from jax.ops import index, index_update
import ipdb
import scipy


(coils, ma, currents) = get_ncsx_data()
cc = CoilCollection(coils, currents, ma.nfp, True)
bs = BiotSavart(cc.coils, cc.currents)


### SET UP SURFACE ###
ss = 1
Nphi   = 11
Ntheta = 11

# you always need an ODD number of collocation points in the (phi,theta) directions, otherwise
# the differentiation matrix will NOT be invertible.
if ma.nfp*Nphi % 2 == 0:
    raise Exception("Sorry, even nfp's not implemented yet\n")
phi   = np.linspace(0, 1./ma.nfp,       Nphi, endpoint = False)
if ss == 1:
    theta = np.linspace(0, 1.       , 2*Ntheta-1, endpoint = False)
    theta = theta[:Ntheta]
else:
    theta = np.linspace(0, 1.       , Ntheta, endpoint = False)

dofs = np.zeros( (Nphi,Ntheta,3) )
r = 0.15
phi_grid, theta_grid = np.meshgrid( phi, theta )
phi_grid = phi_grid.T
theta_grid = theta_grid.T

# rotate the surface in the + or - cylindrical phi direction
flip_phi = -1
ma_xyz = stelleratorsymmetriccylindricalfouriercurve_pure(ma.get_dofs(), flip_phi * phi_grid[:,0], ma.order, ma.nfp)
R = jnp.sqrt(ma_xyz[:,0]**2 + ma_xyz[:,1]**2)
R = jnp.tile( R[:,None], (1,Ntheta) )
Z = jnp.tile( ma_xyz[:,2][:,None], (1,Ntheta) )

# you can also change in which direction theta increases on the surface (CW, or CCW)
flip_theta = 1
dofs[:,:,0] =  ( R  + r * np.cos(flip_theta * 2. * np.pi * theta_grid) ) *  np.cos(flip_phi * 2. * np.pi * phi_grid)
dofs[:,:,1] =  ( R  + r * np.cos(flip_theta * 2. * np.pi * theta_grid) ) * -np.sin(flip_phi * 2. * np.pi * phi_grid)
dofs[:,:,2] =  Z  - r * np.sin(  flip_theta * 2. * np.pi * theta_grid)
dofs = dofs.flatten()
xyzi = np.concatenate( (dofs, np.array([-0.39]) ) )

R_major = 1.5
label = (2 * np.pi * R_major) * (2 * np.pi * r)

# the convergence of the solver really depends on (flip_phi, flip_theta) and if it doesn't converge
# sometimes changing the sign from +/- helps
########################


surf = JaxCartesianMagneticSurface( phi, theta , ma.nfp, ss, flip_phi, label, bs, cc)
surf.set_dofs(xyzi)
surf.plot(apply_symmetries = True, closed_loop = True)


# you can also interpolate onto a finer grid here
surf_fine = surf.interpolated_surface(13,13)
surf = JaxCartesianMagneticSurface(surf_fine, bs, cc, label, surf.iota )
surf.updateBoozer()
surf.plot(apply_symmetries = True, closed_loop = True)
