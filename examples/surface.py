import numpy as np
import scipy.linalg as sp

#A Class for storing surface meta data. 
class Surface:

	def __init__(self, nptheta, npvarphi, nfp):
		#Number of points used to discretize the surface
		self.np_theta = nptheta
		self.np_varphi = npvarphi
		
		#Number of field periods
		self.nfp = nfp
		
		#Calculate spectral derivative matrix based on Matt Landreman's sfincs code:
		
	#A direct python adaptation of Matt Landreman's sfincs code for "Scheme 20"
	def generate_diff_matrix(self, N, xMin, xMax):
		h = 2 * np.pi / N   #grid spacing
		kk = np.linspace(1, N - 1, N - 1)
		
		n1 = int(np.floor((N - 1)/2))
		n2 = int(np.ceil((N - 1)/2))
		
		if N % 2 == 0:
			topc = 1 / (np.tan(np.linspace(1, n2, n2) * (h/2)))
			col1 = np.concatenate((np.array([0]), 0.5 * np.power(-1, kk) * np.concatenate((topc, -np.flip(topc[0:n1])))))
			col1[int(N/2)] = 0 #force this element to zero to avoid weird floating point issues
			
		else:
			topc = 1 / (np.sin(np.linspace(1, n2, n2) * (h/2)))
			col1 = np.concatenate((np.array([0]), 0.5 * np.power(-1, kk) * np.concatenate((topc, np.flip(topc[0:n1])))))
			
		row1 = -col1
		D = (2 * np.pi)/(xMax - xMin) * sp.toeplitz(col1, row1)
		return D
