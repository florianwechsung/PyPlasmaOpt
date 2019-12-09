import os
import numpy as np
from .curve import CartesianFourierCurve, StelleratorSymmetricCylindricalFourierCurve

def get_matt_data(Nt=4, nfp=2, ppp=10):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    outcoils_matt = np.loadtxt(os.path.join(dir_path, "..", "data", "outcoils_matt.dat"), delimiter=',')
    num_coils = 6
    coils = [CartesianFourierCurve(6, np.linspace(0, 1, Nt*ppp, endpoint=False)) for i in range(num_coils)]
    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = outcoils_matt[0, 6*ic + 1]
        coils[ic].coefficients[1][0] = outcoils_matt[0, 6*ic + 3]
        coils[ic].coefficients[2][0] = outcoils_matt[0, 6*ic + 5]
        for io in range(0, Nt-1):
            coils[ic].coefficients[0][2*io+1] = outcoils_matt[io+1, 6*ic + 0]
            coils[ic].coefficients[0][2*io+2] = outcoils_matt[io+1, 6*ic + 1]
            coils[ic].coefficients[1][2*io+1] = outcoils_matt[io+1, 6*ic + 2]
            coils[ic].coefficients[1][2*io+2] = outcoils_matt[io+1, 6*ic + 3]
            coils[ic].coefficients[2][2*io+1] = outcoils_matt[io+1, 6*ic + 4]
            coils[ic].coefficients[2][2*io+2] = outcoils_matt[io+1, 6*ic + 5]
        coils[ic].update()

    ma = StelleratorSymmetricCylindricalFourierCurve(4, nfp, np.linspace(0, 1, Nt*ppp, endpoint=False))
    ma.coefficients[0][0] = 1.
    ma.coefficients[0][1] = 0.076574
    ma.coefficients[0][2] = 0.0032607
    ma.coefficients[0][3] = 2.5405e-05

    ma.coefficients[1][0] = -0.07605
    ma.coefficients[1][1] = -0.0031845
    ma.coefficients[1][2] = -3.1852e-05
    ma.update()
    return (coils, ma)
