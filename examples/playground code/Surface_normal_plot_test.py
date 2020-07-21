from ComputeBoozer import compute_boozer, cross
from scipy.optimize import newton
import numpy as np
from pyplasmaopt import *
import surface as s

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')

nfp = 2
N = 20

(coils, ma) = get_matt_data(nfp=2, ppp=20, at_optimum=True)

currents = [1e5 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
eta_bar = -2.105800979374183

stellarator = CoilCollection(coils, currents, nfp, True)


#making a toroidal initial guess
#remove final endpoint because of periodicity
#Phi direction
u = np.linspace(0, 2*np.pi / nfp, N + 1)
#theta direction
v = np.linspace(0, 2*np.pi, N + 1)

u = u[:-1]
v = v[:-1]

surface_meta = s.Surface(N, N, nfp)

#Major radius
c = 1
#Minor radius
a = 0.4
#Compute surface area of toroidal initial guess
sa_target = 4 * np.pi**2 * c * a

U, V = np.meshgrid(u, v)

#X = (c + a * np.cos(V))*np.cos(U)
#Y = -(c + a * np.cos(V))*np.sin(U)
#Z = - a * np.sin(V)

X = np.loadtxt("x_coord.txt", delimiter=',')
Y = np.loadtxt("y_coord.txt", delimiter=',')
Z = np.loadtxt("z_coord.txt", delimiter=',')
#import ipdb; ipdb.set_trace(context=21) 

iota_init = 2.25

xtemp = X[:, :, np.newaxis]
ytemp = Y[:, :, np.newaxis]
ztemp = Z[:, :, np.newaxis]
	
x_vec_mat = np.concatenate((xtemp, ytemp, ztemp), 2)

def rotate_z(input_mat, nfp):
	out_mat = input_mat
	for i in range(1, nfp):
		rot_mat = np.array([[np.cos( i * 2 * np.pi / nfp), -np.sin(i * 2 * np.pi / nfp), 0], [np.sin(i * 2 * np.pi / nfp), np.cos(i * 2 * np.pi / nfp), 0], [0, 0, 1]])
		out_mat = np.concatenate((out_mat, input_mat @ rot_mat), 1)
	return out_mat

Vec_rot = rotate_z(x_vec_mat, nfp)

X_rot = Vec_rot[:, :, 0]
Y_rot = Vec_rot[:, :, 1]
Z_rot = Vec_rot[:, :, 2]

#Compute spectral derivative matrices

D = surface_meta.generate_diff_matrix(surface_meta.np_theta, 0, 2 * np.pi)
D_phi = surface_meta.generate_diff_matrix(surface_meta.np_theta * nfp, 0, 2 * np.pi)

D_phi = D_phi[0:N, :]

#Apply the differentation matrix to the matrices of each cartesian component, and then rotate around the z axis
#import ipdb; ipdb.set_trace(context=21) 
x_dvarphi = X_rot @ np.transpose(D_phi) 
y_dvarphi = Y_rot @ np.transpose(D_phi) 
z_dvarphi = Z_rot @ np.transpose(D_phi) 

x_dphi_temp = x_dvarphi[:, :, np.newaxis]	
y_dphi_temp = y_dvarphi[:, :, np.newaxis]	
z_dphi_temp = z_dvarphi[:, :, np.newaxis]	

phi_vec = np.concatenate((x_dphi_temp, y_dphi_temp, z_dphi_temp), 2)


x_dtheta = D @ X
y_dtheta = D @ Y
z_dtheta = D @ Z

x_dtheta_temp = x_dtheta[:, :, np.newaxis]	
y_dtheta_temp = y_dtheta[:, :, np.newaxis]	
z_dtheta_temp = z_dtheta[:, :, np.newaxis]	

theta_vec = np.concatenate((x_dtheta_temp, y_dtheta_temp, z_dtheta_temp), 2)

normal_x = cross(y_dtheta, z_dtheta, y_dvarphi, z_dvarphi)
normal_y = cross(z_dtheta, x_dtheta, z_dvarphi, x_dvarphi)
normal_z = cross(x_dtheta, y_dtheta, x_dvarphi, y_dvarphi)

ax.plot_surface(X_rot, Y_rot, Z_rot, cmap=cm.plasma, linewidth=0, antialiased=True)
ax.quiver(X, Y, Z, -normal_x, -normal_y, -normal_z, length=0.1, normalize=True)
ax.quiver(X, Y, Z, phi_vec[:, :, 0], phi_vec[:, :, 1], phi_vec[:, :, 2], length=0.1, normalize=True, color='g')
ax.quiver(X, Y, Z, theta_vec[:, :, 0], theta_vec[:, :, 1], theta_vec[:, :, 2], length=0.1, normalize=True, color='r')

for coil in coils:
	coil.plot(ax=ax, show=False)

ax.set_zlim(-1, 1)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)



plt.show()




