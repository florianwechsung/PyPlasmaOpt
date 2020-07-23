from ComputeBoozer import compute_boozer
from scipy import optimize
import numpy as np
from pyplasmaopt import *
import surface as s
import time

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

	X_flat = X.flatten()
	Y_flat = Y.flatten()
	Y_flat = Y_flat[1:]
	Z_flat = Z.flatten()
	Z_flat = Z_flat[1:]
	iota_init = -0.159453462942961

	#import ipdb; ipdb.set_trace(context=21)
	#Flattening all of the surface information and rotational transform into one very long 1-D array
	x = np.concatenate((X_flat, Y_flat, Z_flat, np.array([iota_init])))

	def booz_call(x, f):
		print(np.max(np.abs(f)))

	compute_boozer(x, surface_meta, cc, sa_target)

	return

	sol = optimize.root(compute_boozer, x, args=(surface_meta, cc, sa_target), jac=True, method='lm', options={'col_deriv': 0, 'xtol': 1.49012e-08, 'ftol': 1.49012e-08, 'gtol': 0.0, 'maxiter': 25, 'eps': 0.0, 'factor': 0.1, 'diag': None}) 

	print("Optimization successful?", sol.success)

	#Reshape optimize.root output to be able to plot
	total_points = surface_meta.np_theta * surface_meta.np_varphi

	x_sol = sol.x[0:total_points]
	y_sol = sol.x[total_points: 2*total_points]
	z_sol = sol.x[2*total_points:3*total_points]
	iota_sol = sol.x[3*total_points]

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




