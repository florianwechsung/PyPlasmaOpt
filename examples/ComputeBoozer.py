from surface import *
import numpy as np
from pyplasmaopt import *

def cross(a1, a2, b1, b2):
	return  a1 * b2 - a2 * b1

def cross1(ax, ay, az, bx, by, bz):
	return ay * bz - az * by
	
def cross2(ax, ay, az, bx, by, bz):
	return az * bx - ax * bz

def cross3(ax, ay, az, bx, by, bz):
	return ax * by - ay * bx

def compute_boozer(x_flat, surface_meta, coilCollection, sa_target):
	total_points = surface_meta.np_theta * surface_meta.np_varphi
	
	import ipdb; ipdb.set_trace(context=21)
	x_array = x_flat[0:total_points]
	y_array = x_flat[total_points: 2*total_points - 1]
	z_array = x_flat[2*total_points:3*total_points - 2]
	iota = x_flat[3*total_points - 2]

	#Initializing matrices to store the cartesian component of each surface point. 
	#Each of these matrices stores one of the three cartesian components associated with each point on the surface. By applying the spectral derivative matrix we take the derivative of this coordinate with respect to theta or phi
	
	xpos = np.reshape(x_array, (surface_meta.np_theta, surface_meta.np_varphi))
	ypos = np.reshape(y_array, (surface_meta.np_theta, surface_meta.np_varphi))
	zpos = np.reshape(z_array, (surface_meta.np_theta, surface_meta.np_varphi))
	
	#Construction of a matrix of 3 vectors 
	#Add an additional dimension to each of the NxN matrices of vector componenets, and then concatenate the matrices along this new axis:
	xtemp = xpos[:, :, np.newaxis]
	ytemp = ypos[:, :, np.newaxis]
	ztemp = zpos[:, :, np.newaxis]
	
	x_vector_matrix = np.concatenate((xtemp, ytemp, ztemp), 2)
	
	vec_mat_rot = x_vector_matrix
	for i in range(1, surface_meta.nfp):
		rot_mat = np.array([[np.cos( i * 2 * np.pi / surface_meta.nfp), -np.sin(i * 2 * np.pi / surface_meta.nfp), 0], [np.sin(i * 2 * np.pi / surface_meta.nfp), np.cos(i * 2 * np.pi / surface_meta.nfp), 0], [0, 0, 1]])
		vec_mat_rot = np.concatenate((vec_mat_rot, x_vector_matrix @ rot_mat), 1)
	
	#breaking down new rotated vector matrix into scalar matrices of cartesian components
	X_rot = vec_mat_rot[:, :, 0]
	Y_rot = vec_mat_rot[:, :, 1]
	Z_rot = vec_mat_rot[:, :, 2]
	
	#Compute magnetic field and field grad at every point on the initial surface guess (x_init)
	bs = BiotSavart(coilCollection.coils, coilCollection.currents)
	#Does set points require a linearized array of points? Yes
	x_vector_as_array = np.reshape(x_vector_matrix, (total_points, 3))
	bs.set_points(x_vector_as_array)
	bs.compute(bs.points)
	
	#Take the list of magnetic field vectors returned by biot savart and reshape them back into the matrix form of the surface input. 
	B_Matrix = np.reshape(bs.B, (surface_meta.np_theta, surface_meta.np_varphi, 3))
	
	#A Matrix of the magnetic field vector magnitudes at each point on the surface.
	B_mag2 = B_Matrix[:, :, 0]**2 + B_Matrix[:, :, 1]**2 + B_Matrix[:, :, 2]**2
	
	#Compute spectral derivative matrices
	D_thet = surface_meta.generate_diff_matrix(surface_meta.np_theta, 0, 2 * np.pi)
	D_phi = surface_meta.generate_diff_matrix(surface_meta.np_theta * surface_meta.nfp, 0, 2 * np.pi)

	#Only select the portion of the D_phi matrix relevant to one field period
	D_phi = D_phi[0:surface_meta.np_theta, :]
	
	#Apply the differentation matrix to the matrices of each cartesian component
	x_dvarphi = X_rot @ np.transpose(D_phi) 
	y_dvarphi = Y_rot @ np.transpose(D_phi) 
	z_dvarphi = Z_rot @ np.transpose(D_phi) 
	
	x_dtheta = D_thet @ xpos
	y_dtheta = D_thet @ ypos
	z_dtheta = D_thet @ zpos
	
	#Compute a matrix of the magnitude of the surface normal at each point on the surface
	normal_x = -cross(y_dtheta, z_dtheta, y_dvarphi, z_dvarphi)
	normal_y = -cross(z_dtheta, x_dtheta, z_dvarphi, x_dvarphi)
	normal_z = -cross(x_dtheta, y_dtheta, x_dvarphi, y_dvarphi)
	
	normal_mag = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
	
	#Compute RHS of equation 17 from Andrew's Simon's slides for each component:

	#Compute constant G
	G = 2 * surface_meta.nfp * np.sum(coilCollection._base_currents) * (4 * np.pi * 10**(-7) / (2 * np.pi))
	
	residual_x = B_Matrix[:, :, 0] - ((B_mag2[:, :] / G) * (x_dvarphi + (iota * x_dtheta)))
	residual_y = B_Matrix[:, :, 1] - ((B_mag2[:, :] / G) * (y_dvarphi + (iota * y_dtheta)))
	residual_z = B_Matrix[:, :, 2] - ((B_mag2[:, :] / G) * (z_dvarphi + (iota * z_dtheta)))
	
	rhs_x = (B_mag2 / G) * (x_dvarphi + (iota * x_dtheta))
	rhs_y = (B_mag2 / G) * (y_dvarphi + (iota * y_dtheta))
	rhs_z = (B_mag2 / G) * (z_dvarphi + (iota * z_dtheta))
	
	residual_sa = (4*np.pi**2)/(surface_meta.np_theta * surface_meta.np_varphi) * np.sum(np.sum(normal_mag)) - sa_target;
	
	residuals = np.concatenate((residual_x.flatten(), residual_y.flatten(), residual_z.flatten(), residual_sa.flatten()))
	assert residuals.shape == x_flat.shape
	
	#Compute Jacobian:
	N = surface_meta.np_theta * surface_meta.nfp * surface_meta.np_varphi
	N2 = surface_meta.np_theta * surface_meta.np_varphi
	Rot = np.zeros((3*surface_meta.nfp, 3))
	
	nfp = surface_meta.nfp
	
	for t in range(0, nfp):
		Rot[t] = np.array([np.cos(t * 2 * np.pi / nfp), np.sin(t * 2 * np.pi / nfp), 0])
		Rot[nfp + t] = np.array([-np.sin(t * 2 * np.pi / nfp), np.cos(t * 2 * np.pi / nfp), 0])
		Rot[2 * nfp + t] = np.array([0, 0, 1])
	
	rotated_matrix = np.kron(Rot, np.eye(surface_meta.np_theta * surface_meta.np_varphi))
	
	dvarphik1 = np.kron(D_phi, np.eye(surface_meta.np_theta, surface_meta.np_varphi)) @ rotated_matrix[0:N, :]
	dvarphik2 = np.kron(D_phi, np.eye(surface_meta.np_theta, surface_meta.np_varphi)) @ rotated_matrix[N:2*N, :]
	dvarphik3 = np.kron(D_phi, np.eye(surface_meta.np_theta, surface_meta.np_varphi)) @ rotated_matrix[2*N:3*N, :]
	
	dthetak = np.kron(np.eye(surface_meta.np_theta, surface_meta.np_varphi), D_thet)
	
	zero = np.zeros(surface_meta.np_theta * surface_meta.np_varphi)
	dB_dX_Matrix = np.reshape(bs.dB_by_dX, (surface_meta.np_theta, surface_meta.np_varphi, 3, 3))
	
	B_mag2_dx = 2*B_Matrix[:, :, 0] * dB_dX_Matrix[:, :, 0, 0] + 2*B_Matrix[:, :, 1] * dB_dX_Matrix[:, :, 1, 0] + 2*B_Matrix[:, :, 2] * dB_dX_Matrix[:, :, 2, 0]
	B_mag2_dy = 2*B_Matrix[:, :, 0] * dB_dX_Matrix[:, :, 0, 1] + 2*B_Matrix[:, :, 1] * dB_dX_Matrix[:, :, 1, 1] + 2*B_Matrix[:, :, 2] * dB_dX_Matrix[:, :, 2, 1]
	B_mag2_dz = 2*B_Matrix[:, :, 0] * dB_dX_Matrix[:, :, 0, 2] + 2*B_Matrix[:, :, 1] * dB_dX_Matrix[:, :, 1, 2] + 2*B_Matrix[:, :, 2] * dB_dX_Matrix[:, :, 2, 2]
	
	B_mag2_dx_vec = np.reshape(B_mag2_dx, (N2, 1), order='F')
	B_mag2_dy_vec = np.reshape(B_mag2_dy, (N2, 1), order='F')
	B_mag2_dz_vec = np.reshape(B_mag2_dz, (N2, 1), order='F')
	
	B_mag2_vec = np.reshape(B_mag2, (N2, 1), order='F')
	
	x_dtheta_vec = np.reshape(x_dtheta, (N2, 1), order='F')
	y_dtheta_vec = np.reshape(y_dtheta, (N2, 1), order='F')
	z_dtheta_vec = np.reshape(z_dtheta, (N2, 1), order='F')
	
	x_dvarphi_vec = np.reshape(x_dvarphi, (N2, 1), order='F')
	y_dvarphi_vec = np.reshape(y_dvarphi, (N2, 1), order='F')
	z_dvarphi_vec = np.reshape(z_dvarphi, (N2, 1), order='F')
	
	dn = np.zeros((3, 3, N2, N2))
	dn[0, 0] = cross1(dvarphik1[:, 0:N2], dvarphik2[:, 0:N2], dvarphik3[:, 0:N2], x_dtheta_vec, y_dtheta_vec, z_dtheta_vec) + cross1(x_dvarphi_vec, y_dvarphi_vec, z_dvarphi_vec, dthetak, zero, zero)
	dn[1, 0] = cross2(dvarphik1[:, 0:N2], dvarphik2[:, 0:N2], dvarphik3[:, 0:N2], x_dtheta_vec, y_dtheta_vec, z_dtheta_vec) + cross2(x_dvarphi_vec, y_dvarphi_vec, z_dvarphi_vec, dthetak, zero, zero)
	dn[2, 0] = cross3(dvarphik1[:, 0:N2], dvarphik2[:, 0:N2], dvarphik3[:, 0:N2], x_dtheta_vec, y_dtheta_vec, z_dtheta_vec) + cross3(x_dvarphi_vec, y_dvarphi_vec, z_dvarphi_vec, dthetak, zero, zero)
	
	dn[0, 1] = cross1(dvarphik1[:, N2:2*N2], dvarphik2[:, N2:2*N2], dvarphik3[:, N2:2*N2], x_dtheta_vec, y_dtheta_vec, z_dtheta_vec) + cross1(x_dvarphi_vec, y_dvarphi_vec, z_dvarphi_vec, zero, dthetak, zero)
	dn[1, 1] = cross2(dvarphik1[:, N2:2*N2], dvarphik2[:, N2:2*N2], dvarphik3[:, N2:2*N2], x_dtheta_vec, y_dtheta_vec, z_dtheta_vec) + cross2(x_dvarphi_vec, y_dvarphi_vec, z_dvarphi_vec, zero, dthetak, zero)
	dn[2, 1] = cross3(dvarphik1[:, N2:2*N2], dvarphik2[:, N2:2*N2], dvarphik3[:, N2:2*N2], x_dtheta_vec, y_dtheta_vec, z_dtheta_vec) + cross3(x_dvarphi_vec, y_dvarphi_vec, z_dvarphi_vec, zero, dthetak, zero)
	
	dn[0, 2] = cross1(dvarphik1[:, 2*N2:3*N2], dvarphik2[:, 2*N2:3*N2], dvarphik3[:, 2*N2:3*N2], x_dtheta_vec, y_dtheta_vec, z_dtheta_vec) + cross1(x_dvarphi_vec, y_dvarphi_vec, z_dvarphi_vec, zero, zero, dthetak)
	dn[1, 2] = cross2(dvarphik1[:, 2*N2:3*N2], dvarphik2[:, 2*N2:3*N2], dvarphik3[:, 2*N2:3*N2], x_dtheta_vec, y_dtheta_vec, z_dtheta_vec) + cross2(x_dvarphi_vec, y_dvarphi_vec, z_dvarphi_vec, zero, zero, dthetak)
	dn[2, 2] = cross3(dvarphik1[:, 2*N2:3*N2], dvarphik2[:, 2*N2:3*N2], dvarphik3[:, 2*N2:3*N2], x_dtheta_vec, y_dtheta_vec, z_dtheta_vec) + cross3(x_dvarphi_vec, y_dvarphi_vec, z_dvarphi_vec, zero, zero, dthetak)
	
	normal_x_vec = np.reshape(normal_x, (N2, 1), order='F')
	normal_y_vec = np.reshape(normal_y, (N2, 1), order='F')
	normal_z_vec = np.reshape(normal_z, (N2, 1), order='F')
	normal_mag_vec = np.reshape(normal_mag, (N2, 1), order='F')
	
	dnmag = np.zeros((3, N2, N2))
	
	#Something is wrong here, these values do not quite agree with those in the MATLAB
	dnmag[0] = np.divide(normal_x_vec * dn[0, 0] + normal_y_vec * dn[1, 0] + normal_z_vec * dn[2, 0], normal_mag_vec)
	dnmag[1] = (normal_x_vec * dn[0, 1] + normal_y_vec * dn[1, 1] + normal_z_vec * dn[2, 1]) / normal_mag_vec
	dnmag[2] = (normal_x_vec * dn[0, 2] + normal_y_vec * dn[1, 2] + normal_z_vec * dn[2, 2]) / normal_mag_vec
	
	dB_dX_x_x_vec = np.reshape(dB_dX_Matrix[:, :, 0, 0], (N2, 1), order='F')
	dB_dX_x_y_vec = np.reshape(dB_dX_Matrix[:, :, 0, 1], (N2, 1), order='F')
	dB_dX_x_z_vec = np.reshape(dB_dX_Matrix[:, :, 0, 2], (N2, 1), order='F')
	
	dB_dX_y_x_vec = np.reshape(dB_dX_Matrix[:, :, 1, 0], (N2, 1), order='F')
	dB_dX_y_y_vec = np.reshape(dB_dX_Matrix[:, :, 1, 1], (N2, 1), order='F')
	dB_dX_y_z_vec = np.reshape(dB_dX_Matrix[:, :, 1, 2], (N2, 1), order='F')
	
	dB_dX_z_x_vec = np.reshape(dB_dX_Matrix[:, :, 2, 0], (N2, 1), order='F')
	dB_dX_z_y_vec = np.reshape(dB_dX_Matrix[:, :, 2, 1], (N2, 1), order='F')
	dB_dX_z_z_vec = np.reshape(dB_dX_Matrix[:, :, 2, 2], (N2, 1), order='F')
	
	#RHS 1
	rhs1_dx = dB_dX_x_x_vec - (B_mag2_dx_vec / G)*(x_dvarphi_vec + iota * x_dtheta_vec)
	rhs1_dx = np.diag(rhs1_dx.flatten())
	rhs1_dx = rhs1_dx - (B_mag2_vec/G)*(dvarphik1[:, 0:N2] + iota * dthetak)
	
	rhs1_dy = dB_dX_x_y_vec - (B_mag2_dy_vec / G)*(x_dvarphi_vec + iota * x_dtheta_vec)
	rhs1_dy = np.diag(rhs1_dy.flatten())
	rhs1_dy = rhs1_dy - (B_mag2_vec/G)*dvarphik1[:, N2:2*N2]
	
	rhs1_dz = dB_dX_x_z_vec - (B_mag2_dz_vec / G)*(x_dvarphi_vec + iota * x_dtheta_vec)
	rhs1_dz = np.diag(rhs1_dz.flatten())
	rhs1_dz = rhs1_dz - (B_mag2_vec/G)*dvarphik1[:, 2*N2:3*N2]
	
	rhs1_diota = -(B_mag2_vec/G)*x_dtheta_vec
	
	rhs1_J = np.concatenate((rhs1_dx, rhs1_dy, rhs1_dz, rhs1_diota), 1)

	#RHS 2
	rhs2_dx = dB_dX_y_x_vec - (B_mag2_dx_vec / G)*(y_dvarphi_vec + iota * y_dtheta_vec)
	rhs2_dx = np.diag(rhs2_dx.flatten())
	rhs2_dx = rhs2_dx - (B_mag2_vec/G)*dvarphik2[:, 0:N2]
	
	rhs2_dy = dB_dX_y_y_vec - (B_mag2_dy_vec / G)*(y_dvarphi_vec + iota * y_dtheta_vec)
	rhs2_dy = np.diag(rhs2_dy.flatten())
	rhs2_dy = rhs2_dy - (B_mag2_vec/G)*(dvarphik2[:, N2:2*N2] + iota * dthetak)
	
	rhs2_dz = dB_dX_y_z_vec - (B_mag2_dz_vec / G)*(y_dvarphi_vec + iota * y_dtheta_vec)
	rhs2_dz = np.diag(rhs2_dz.flatten())
	rhs2_dz = rhs2_dz - (B_mag2_vec/G)*dvarphik2[:, 2*N2:3*N2]
	
	rhs2_diota = -(B_mag2_vec/G)*y_dtheta_vec
	
	rhs2_J = np.concatenate((rhs2_dx, rhs2_dy, rhs2_dz, rhs2_diota), 1)
	
	#RHS 3
	rhs3_dx = dB_dX_z_x_vec - (B_mag2_dx_vec / G)*(z_dvarphi_vec + iota * z_dtheta_vec)
	rhs3_dx = np.diag(rhs3_dx.flatten())
	rhs3_dx = rhs3_dx - (B_mag2_vec/G)*dvarphik3[:, 0:N2]
	
	rhs3_dy = dB_dX_z_y_vec - (B_mag2_dy_vec / G)*(z_dvarphi_vec + iota * z_dtheta_vec)
	rhs3_dy = np.diag(rhs3_dy.flatten())
	rhs3_dy = rhs3_dy - (B_mag2_vec/G)*dvarphik3[:, N2:2*N2]
	
	rhs3_dz = dB_dX_z_z_vec - (B_mag2_dz_vec / G)*(z_dvarphi_vec + iota * z_dtheta_vec)
	rhs3_dz = np.diag(rhs3_dz.flatten())
	rhs3_dz = rhs3_dz - (B_mag2_vec/G)*(dvarphik3[:, 2*N2:3*N2] + iota * dthetak)
	
	rhs3_diota = -(B_mag2_vec/G)*z_dtheta_vec
	
	rhs3_J = np.concatenate((rhs3_dx, rhs3_dy, rhs3_dz, rhs3_diota), 1)
	
	#RHS 4
	rhs4_dx = (4*np.pi**2)/(surface_meta.np_theta * surface_meta.np_varphi) * np.sum(dnmag[0], 0)
	rhs4_dy = (4*np.pi**2)/(surface_meta.np_theta * surface_meta.np_varphi) * np.sum(dnmag[1], 0)
	rhs4_dz = (4*np.pi**2)/(surface_meta.np_theta * surface_meta.np_varphi) * np.sum(dnmag[2], 0)
	rhs4_diota = 0
	rhs4_J = np.concatenate((rhs4_dx, rhs4_dy, rhs4_dz, np.array([rhs4_diota])))
	rhs4_J = np.reshape(rhs4_J, (1, 3*N2 + 1))
	
	J = np.concatenate((rhs1_J, rhs2_J, rhs3_J, rhs4_J), 0)
	#import ipdb; ipdb.set_trace(context=21)
	
	J = np.delete(J, (400, 800), 1)
	J = np.delete(J, (0, 400), 0)
	
	print(J.shape)
	
	np.savetxt("Py_Jac_2.txt", J)
	
	print("Max residual:", np.amax(np.abs(residuals)))
	
	#print("X residual:", np.sum(residual_x**2))
	#print("Y residual:", np.sum(residual_y**2))
	#print("Z residual:", np.sum(residual_z**2))
	#print("SA residual:", residual_sa)
	#print("Total residual:")
	#print(np.sum(residuals**2))
	#print(residuals)
	
	#from mpl_toolkits.mplot3d import Axes3D
	#import matplotlib.pyplot as plt
	#from matplotlib import cm
	
	#fig = plt.figure()
	#ax = fig.gca(projection='3d')
	
	#ax.plot_surface(xpos, ypos, zpos, cmap=cm.plasma)
	
	#ax.set_zlim(-1, 1)
	#ax.set_xlim(-1.2, 1.2)
	#ax.set_ylim(-1.2, 1.2)

	#plt.show()
	

	return residuals, J
	
	
	
	
	
	
	
	
	
	
	
	
