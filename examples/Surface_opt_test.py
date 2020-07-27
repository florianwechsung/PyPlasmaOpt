from ComputeBoozer import compute_boozer
from scipy import optimize
import numpy as np
from pyplasmaopt import *
import surface as s
import time
from generate_toroidal_guess import toroidal_guess


def main():
	nfp = 2
	N = 20

	#Imports coil data
	(coils, ma) = get_matt_data(nfp=2, ppp=20, at_optimum=True)

	#Currents given by Andrew
	currents = np.array([-219001.804742907, -215024.346505962, -212678.904950201, -213524.933314340, -205978.356488100, -198316.662919875])
	eta_bar = -2.105800979374183

	cc = CoilCollection(coils, currents, nfp, True)

	surface_meta = s.Surface(N, N, nfp)

	X = np.loadtxt("x_coord.txt", delimiter=',')
	Y = np.loadtxt("y_coord.txt", delimiter=',')
	Z = np.loadtxt("z_coord.txt", delimiter=',')

	sa_target = 7.895683520871486

	#X, Y, Z, sa_target = toroidal_guess(1.5, .2, N, nfp)
	
	X_flat = X.flatten('F')
	Y_flat = Y.flatten('F')
	Y_flat = Y_flat[1:]
	Z_flat = Z.flatten('F')
	Z_flat = Z_flat[1:]
	iota_init = -0.159453462942961

	#import ipdb; ipdb.set_trace(context=21)
	#Flattening all of the surface information and rotational transform into one very long 1-D array
	x = np.concatenate((X_flat, Y_flat, Z_flat, np.array([iota_init])))

	def iter_newton(X, imax = 1e6,tol = 1e-5):
   		for i in range(int(imax)):
   			Y, J = compute_boozer(X, surface_meta, cc, sa_target)
   			print(np.amax(np.abs(Y)))
   			res_prev = np.amax(np.abs(Y))
   			dX = np.linalg.solve(J,Y) # solve for increment from JdX = Y 
   			X_p = X - dX # step X by dX 
   			
   			Y_p, J_p = compute_boozer(X_p, surface_meta, cc, sa_target)
   			
   			while np.amax(np.abs(Y_p)) > res_prev:
   				dX = dX/2
   				X_p = X-dX 
   				
   				Y_p, J_p = compute_boozer(X_p, surface_meta, cc, sa_target)
   				
   			X = X_p
   			
   			if np.linalg.norm(dX)<tol: # break if converged
   				print('converged.')
   				break
   				
   		return X 

	sol = iter_newton(x)
	
	return


	#Reshape optimize.root output to be able to plot
	total_points = surface_meta.np_theta * surface_meta.np_varphi

	x_sol = np.zeros(total_points)
	y_sol = np.zeros(total_points)
	z_sol = np.zeros(total_points)

	x_sol = sol.x[0:total_points]
	y_sol[1:] = sol.x[total_points: 2*total_points - 1]
	z_sol[1:] = sol.x[2*total_points - 1:3*total_points - 2]
	iota_sol = sol.x[3*total_points - 2]

	x_sol = np.reshape(x_sol, X.shape)
	y_sol = np.reshape(y_sol, X.shape)
	z_sol = np.reshape(z_sol, X.shape)

	#Plot the resulting surface and the initial surface
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	from matplotlib import cm
	
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	
	ax.plot_surface(x_sol, y_sol, z_sol, cmap=cm.plasma)
	ax.plot_surface(X, Y, Z, cmap=cm.cool)
	
	ax.set_zlim(-1, 1)
	ax.set_xlim(-1.2, 1.2)
	ax.set_ylim(-1.2, 1.2)

	plt.show()

	print(np.amax(X - x_sol))
	print(np.amax(Y - y_sol))
	print(np.amax(Z - z_sol))

if __name__ == '__main__':
	main()




