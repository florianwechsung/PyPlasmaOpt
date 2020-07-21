from surface import *
import numpy as np
from pyplasmaopt import *

def cross(a1, a2, b1, b2):
	return  a1 * b2 - a2 * b1

def compute_boozer(x_flat, surface_meta, coilCollection, sa_target):
	total_points = surface_meta.np_theta * surface_meta.np_varphi
	
	x_array = x_flat[0:total_points]
	y_array = x_flat[total_points: 2*total_points]
	z_array = x_flat[2*total_points:3*total_points]
	iota = x_flat[3*total_points]

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
	normal_x = cross(y_dtheta, z_dtheta, y_dvarphi, z_dvarphi)
	normal_y = cross(z_dtheta, x_dtheta, z_dvarphi, x_dvarphi)
	normal_z = cross(x_dtheta, y_dtheta, x_dvarphi, y_dvarphi)
	
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
	
	#print("X residual:", np.sum(residual_x**2))
	#print("Y residual:", np.sum(residual_y**2))
	#print("Z residual:", np.sum(residual_z**2))
	#print("SA residual:", residual_sa)
	#print("Total residual:")
	#print(np.sum(residuals**2))
	#print(residuals)

	return residuals
	
	
	
	
	
	
	
	
	
	
	
	
