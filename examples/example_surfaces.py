from pyplasmaopt import *
from simsgeo import JaxStelleratorSymmetricCylindricalFourierCurve, StelleratorSymmetricCylindricalFourierCurve, \
FourierCurve, JaxFourierCurve, RotatedCurve, JaxCartesianSurface, stelleratorsymmetriccylindricalfouriercurve_pure
import jax.numpy as jnp
from jax.ops import index, index_update
import ipdb
import scipy


def toroidal_flux(bs, surf):
    points = surf.get_dofs().reshape( (surf.numquadpoints_phi, surf.numquadpoints_theta,3) )[0,:,:]
    bs.set_points(points)
    A = bs.A
    tf = np.mean(np.sum(A * surf.gammadash2(),axis=1) )
    return tf





def func(xyzi, surface, coilCollection, bs, sa_target):
    surface.set_dofs( xyzi[:-1] )
    i = xyzi[-1]
    f,df = boozer( i, surface, coilCollection, bs, sa_target)
    return f,df

def boozer(i, surface, coilCollection, bs, sa_target):            
    
    xyz = surface.get_dofs().reshape( (-1,3) )
    G = 2. * np.pi * jnp.sum( jnp.array( coilCollection._base_currents ) ) * 2. * surface.nfp * (4 * np.pi * 10**(-7) / (2 * np.pi))

    bs.set_points(xyz)
    bs.compute(bs.points)
    B = bs.B.flatten()
    Bmag2     = jnp.tile( (bs.B[:,0]**2 + bs.B[:,1]**2 + bs.B[:,2]**2)[:,None], (1, 3) ).flatten()

    pde     = B - (Bmag2/G) * ( surface.gammadash1().flatten() + i * surface.gammadash2().flatten() )
    sa_cons = jnp.array( [surface.surface_area() - sa_target] )
    rhs = jnp.concatenate( (pde, sa_cons) )

    didx = jnp.arange(xyz.shape[0]) 
    dB_dX = jnp.zeros( (xyz.shape[0], 3, xyz.shape[0], 3) )
    dB_dX = index_update( dB_dX, index[didx, :,didx , :], bs.dB_by_dX ).reshape( (3 * xyz.shape[0], 3 * xyz.shape[0] ) ) 

    

    dBmag2_dX_lin = 2.*(bs.B[:,0][:,None] * bs.dB_by_dX[:,0,:] \
                      + bs.B[:,1][:,None] * bs.dB_by_dX[:,1,:] \
                      + bs.B[:,2][:,None] * bs.dB_by_dX[:,2,:]).reshape( (-1,3) )
    dBmag2_dX = jnp.zeros( (xyz.shape[0], 3, xyz.shape[0], 3) )
    dBmag2_dX = index_update( dBmag2_dX, index[didx,0,didx,:], dBmag2_dX_lin ) 
    dBmag2_dX = index_update( dBmag2_dX, index[didx,1,didx,:], dBmag2_dX_lin ) 
    dBmag2_dX = index_update( dBmag2_dX, index[didx,2,didx,:], dBmag2_dX_lin ) 
    dBmag2_dX = dBmag2_dX.reshape(  (3 * xyz.shape[0], 3 * xyz.shape[0] ) )

    term1 =  (dBmag2_dX/G) * ( surface.gammadash1() + i * surface.gammadash2() ).reshape( (-1,1) )
    term2 = (Bmag2  / G ).reshape( (-1,1) ) * ( surface.Dphi + i * surface.Dtheta )
    dpde_dX = dB_dX - term1 - term2 
    dpde_di = (  -(Bmag2/G) * surface.gammadash2().flatten() ).reshape( (-1,1) )

    sa_cons_dX = surface.surface_area_dX( xyz ).reshape( (1,-1) )
    sa_cons_di = np.array([[0.]])

    drhs_pde = jnp.hstack( (dpde_dX, dpde_di) )
    drhs_cons = jnp.hstack( (sa_cons_dX, sa_cons_di) )
    drhs = jnp.vstack( (drhs_pde, drhs_cons) )

    rhs = index_update(rhs, index[0], rhs[0] + rhs[1] + rhs[2] )
    drhs = index_update(drhs, index[0,:], drhs[0,:] + drhs[1,:] + drhs[2,:] )
    
    keep = np.concatenate( (np.array([0]), np.arange(3,rhs.size) ) )
    rhs = rhs[keep]
    drhs = drhs[keep,:]
    drhs = drhs[:,keep]
    

    return rhs,drhs


(coils, ma, currents) = get_ncsx_data()
cc = CoilCollection(coils, currents, ma.nfp, True)
bs = BiotSavart(cc.coils, cc.currents)



Nphi   = 13
Ntheta = 13
phi   = np.linspace(0, 1./ma.nfp,       Nphi, endpoint = False)
theta = np.linspace(0, 1.       , 2*Ntheta-1, endpoint = False)
theta = theta[:Ntheta]

dofs = np.zeros( (Nphi,Ntheta,3) )
r = 0.2
phi_grid, theta_grid = np.meshgrid( phi, theta )
phi_grid = phi_grid.T
theta_grid = theta_grid.T

# rotate phi in the + or - cylindrical phi direction
flip = -1
ss = 1
ma_xyz = stelleratorsymmetriccylindricalfouriercurve_pure(ma.get_dofs(), flip * phi_grid[:,0], ma.order, ma.nfp)
R = jnp.sqrt(ma_xyz[:,0]**2 + ma_xyz[:,1]**2)
R = jnp.tile( R[:,None], (1,Ntheta) )
Z = jnp.tile( ma_xyz[:,2][:,None], (1,Ntheta) )

dofs[:,:,0] =  ( R  + r * np.cos(2. * np.pi * theta_grid) ) *  np.cos(flip * 2. * np.pi * phi_grid)
dofs[:,:,1] =  ( R  + r * np.cos(2. * np.pi * theta_grid) ) * -np.sin(flip * 2. * np.pi * phi_grid)
dofs[:,:,2] =  Z  - r * np.sin(2. * np.pi * theta_grid)
dofs = dofs.flatten()

surf = JaxCartesianSurface( phi, theta , ma.nfp, ss, flip)
surf.set_dofs(dofs)

sa_target = surf.surface_area()
xyzi = np.concatenate( (dofs, np.array([-0.39]) ) )

print("initial surface area is ", sa_target)


fdf = lambda x : func(x, surf, cc, bs, sa_target)
#sol = scipy.optimize.root( fdf, xyzi, method = 'lm', jac = True)
#print(sol)
#fdf = lambda x : func(x, surf, cc, bs, sa_target)
diff = 1
count = 0
lamb = 1e-6
rhs,drhs = fdf(xyzi)
print("initial norm is ", np.linalg.norm(rhs) )
while diff > 1e-12:
    rhs,drhs = fdf(xyzi)
    
    update = np.linalg.solve(drhs.T @ drhs + lamb * jnp.eye(drhs.shape[0]), drhs.T @ rhs)
    update = np.concatenate( (np.array([update[0],0,0] ), update[1:]) )
    rhstemp,_ = fdf( xyzi -  update)
    while np.linalg.norm(rhstemp) >  np.linalg.norm(rhs):
        lamb = lamb * 10
        update = np.linalg.solve(drhs.T @ drhs + lamb * jnp.eye(drhs.shape[0]), drhs.T @ rhs)
        update = np.concatenate( (np.array([update[0],0,0] ), update[1:]) )
        rhstemp,_ = fdf(xyzi -  update)
    
    lamb = lamb/10
    xyzi = jnp.array(xyzi-update)
    diff = np.linalg.norm(update)
    count += 1
    
    print("accepted step ", jnp.linalg.norm(rhstemp) , "lambda ", lamb)


print(toroidal_flux(bs, surf) )

#surf = surf.interpolated_surface(31,31)
surf.plot(apply_symmetries=False, closed_loop = False, plot_derivative= False)

#ipdb.set_trace(context=21)
