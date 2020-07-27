import numpy as np

#making a toroidal initial guess
def toroidal_guess(major_r, minor_r, N, nfp):

	u = np.linspace(0, 2*np.pi / nfp, N + 1)
	v = np.linspace(0, 2*np.pi / nfp, N + 1)

	#remove final endpoint because of periodicity
	u = u[:-1]
	v = v[:-1]

	#Compute surface area of toroidal initial guess
	sa = 4 * np.pi**2 * major_r * minor_r

	U, V = np.meshgrid(u, v)

	#Returns NxN matrices of cartesian coordinates corresponing to each point in u, v space
	X = (major_r + minor_r * np.cos(V))*np.cos(U)
	Y = - (major_r + minor_r * np.cos(V))*np.sin(U)
	Z = - minor_r * np.sin(V)
	
	return X, Y, Z, sa
