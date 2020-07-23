import os
import numpy as np
from .curve import CartesianFourierCurve, StelleratorSymmetricCylindricalFourierCurve

#returns a set of coils and a magnetic axis based on some data from Dr. Landreman?
#Arguments need to figured out
def get_matt_data(Nt_coils=3, Nt_ma=3, nfp=2, ppp=10, at_optimum=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if at_optimum:
        coil_data = np.loadtxt(os.path.join(dir_path, "..", "data", "matt_optimal.dat"), delimiter=',')
    else:
        coil_data = np.loadtxt(os.path.join(dir_path, "..", "data", "matt_initial.dat"), delimiter=',')
    num_coils = 6
    coils = [CartesianFourierCurve(Nt_coils, np.linspace(0, 1, (Nt_coils+1)*ppp, endpoint=False)) for i in range(num_coils)]
    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = coil_data[0, 6*ic + 1]
        coils[ic].coefficients[1][0] = coil_data[0, 6*ic + 3]
        coils[ic].coefficients[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, Nt_coils):
            coils[ic].coefficients[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            coils[ic].coefficients[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            coils[ic].coefficients[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            coils[ic].coefficients[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            coils[ic].coefficients[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            coils[ic].coefficients[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].update()

    numpoints = (Nt_ma+1)*ppp
    if numpoints % 2 == 0:
        numpoints += 1
        
    #Important: Returns a curve object (the magnetic axis) in cylidrical fourier representation
    ma = StelleratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False))
    if at_optimum:
        ma.coefficients[0][0] = 0.976141492438223
        ma.coefficients[0][1] = 0.112424048908878
        ma.coefficients[0][2] = 0.008616069597869
        ma.coefficients[0][3] = 0.000481649520639

        ma.coefficients[1][0] = -0.149451871576844
        ma.coefficients[1][1] = -0.008946798078974
        ma.coefficients[1][2] = -0.000540954372519
    else:
        ma.coefficients[0][0] = 1.
        ma.coefficients[0][1] = 0.076574
        ma.coefficients[0][2] = 0.0032607
        ma.coefficients[0][3] = 2.5405e-05

        ma.coefficients[1][0] = -0.07605
        ma.coefficients[1][1] = -0.0031845
        ma.coefficients[1][2] = -3.1852e-05

    ma.update()
    return (coils, ma)

def get_flat_data(Nt=4, nfp=3, ppp=10):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    coil_data = np.loadtxt(os.path.join(dir_path, "..", "data", "flat.dat"), delimiter=',')
    num_coils = 3
    coils = [CartesianFourierCurve(Nt, np.linspace(0, 1, Nt*ppp, endpoint=False)) for i in range(num_coils)]
    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = coil_data[0, 6*ic + 1]
        coils[ic].coefficients[1][0] = coil_data[0, 6*ic + 3]
        coils[ic].coefficients[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, Nt-1):
            coils[ic].coefficients[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            coils[ic].coefficients[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            coils[ic].coefficients[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            coils[ic].coefficients[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            coils[ic].coefficients[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            coils[ic].coefficients[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].update()

    numpoints = Nt*ppp+1 if ((Nt*ppp) % 2 == 0) else Nt*ppp
    ma = StelleratorSymmetricCylindricalFourierCurve(Nt-1, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False))
    ma.coefficients[0][0] = 1.
    ma.coefficients[1][0] = 0.01

    ma.update()
    return (coils, ma)
