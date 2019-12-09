import numpy as np
from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve

coil = CartesianFourierCurve(10, np.linspace(0, 1, 80, endpoint=False))
coil.coefficients[1][0] = 1.
coil.coefficients[1][1] = 0.5
coil.coefficients[2][2] = 0.5
coil.update()


ma = StelleratorSymmetricCylindricalFourierCurve(3, 2, np.linspace(0, 1, 80, endpoint=False))
ma.coefficients[0][0] = 1.
ma.coefficients[0][1] = 0.1
ma.coefficients[1][0] = 0.1
ma.update()

bs = BiotSavart([coil], [1])

points = ma.gamma(np.linspace(0., 1., 10000))


import time
start = time.time()
for i in range(3):
    bs.B(points, use_cpp=True)
end = time.time()
print('Time for B', (end-start)*1000)

start = time.time()
for i in range(3):
    bs.dB_by_dX(points, use_cpp=True)
end = time.time()
print('Time for dB_by_dX', (end-start)*1000)





points = ma.gamma(np.linspace(0., 1., 1000))
start = time.time()
for i in range(3):
    bs.dB_by_dcoilcoeff(points, use_cpp=True)
end = time.time()
print('Time for dB_by_dcoilcoeff', (end-start)*1000)
start = time.time()
for i in range(3):
    bs.dB_by_dcoilcoeff_via_chainrule(points, use_cpp=True)
end = time.time()
print('Time for dB_by_dcoilcoeff_via_chainrule', (end-start)*1000)
start = time.time()
for i in range(3):
    bs.d2B_by_dXdcoilcoeff(points, use_cpp=True)
end = time.time()
print('Time for d2B_by_dXdcoilcoeff', (end-start)*1000)
