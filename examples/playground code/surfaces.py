from pyplasmaopt import *
from problem2_objective import get_objective
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

obj = get_objective()

obj.biotsavart.compute(obj.biotsavart.points)

obj.biotsavart.coil_currents = [100000 for coil in obj.biotsavart.coils]


obj.biotsavart.compute(obj.ma.gamma)

print(obj.biotsavart.coil_currents)
print(obj.biotsavart.B, obj.biotsavart.points)

B = obj.biotsavart.B

#Components of the magnetic field vector
Bu = [point[0] for point in B]
Bv = [point[1] for point in B]
Bw = [point[2] for point in B]

#Components of the location of the magnetic field vector
Bx = [point[0] for point in obj.biotsavart.points]
By = [point[1] for point in obj.biotsavart.points]
Bz = [point[2] for point in obj.biotsavart.points]

resolution = 10;

all_points = []

for x in np.linspace(0, 1, resolution):
	for y in np.linspace(0, 1, resolution):
		for z in np.linspace(0, 1, resolution):
			all_points.append([x, y, z])
			print ("added point: " x, y, z)
			
bs = BiotSavart()

		

fig = plt.figure()
position_ax = fig.add_subplot(111, projection='3d')
vector_ax = ax = fig.add_subplot(111, projection='3d')

#position_ax.plot(Bx, By, Bz)
vector_ax.quiver(Bx, By, Bz, Bu, Bv, Bw, length=.3, arrow_length_ratio=0.02)

plt.show()
