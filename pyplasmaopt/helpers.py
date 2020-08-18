import os
import numpy as np
from .curve import CartesianFourierCurve, StelleratorSymmetricCylindricalFourierCurve

def get_24_coil_data(Nt_coils=3, Nt_ma=3, nfp=2, ppp=10, at_optimum=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if at_optimum:
        coil_data = np.loadtxt(os.path.join(dir_path, "data", "matt_optimal.dat"), delimiter=',')
    else:
        coil_data = np.loadtxt(os.path.join(dir_path, "data", "matt_initial.dat"), delimiter=',')
    num_coils = 6
    coils = [CartesianFourierCurve(Nt_coils, np.linspace(0, 1, (Nt_coils+1)*ppp, endpoint=False)) for i in range(num_coils)]
    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = coil_data[0, 6*ic + 1]
        coils[ic].coefficients[1][0] = coil_data[0, 6*ic + 3]
        coils[ic].coefficients[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, min(Nt_coils, 5)):
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
    ma = StelleratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False))
    if at_optimum:
        macoeff0 = [0.976141492438223, 0.112424048908878, 0.008616069597869, 0.000481649520639]
        macoeff1 = [-0.149451871576844, -0.008946798078974, -0.000540954372519]

        for i in range(min(Nt_ma+1, 4)):
            ma.coefficients[0][i] = macoeff0[i]
        for i in range(min(Nt_ma, 3)):
            ma.coefficients[1][i] = macoeff1[i]

    else:
        macoeff0 = [1., 0.076574, 0.0032607, 2.5405e-05]
        macoeff1 = [-0.07605, -0.0031845, -3.1852e-05]

        for i in range(min(Nt_ma+1, 4)):
            ma.coefficients[0][i] = macoeff0[i]
        for i in range(min(Nt_ma, 3)):
            ma.coefficients[1][i] = macoeff1[i]

    ma.update()

    if at_optimum:
        currents = [1e5 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
        eta_bar = -2.105800979374183
    else:
        currents = 6 * [0.]
        eta_bar = -2.25

    return (coils, currents, ma, eta_bar)

def get_flat_data(Nt=4, nfp=3, ppp=10):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    coil_data = np.loadtxt(os.path.join(dir_path, "data", "flat.dat"), delimiter=',')
    num_coils = 3
    coils = [CartesianFourierCurve(Nt, np.linspace(0, 1, Nt*ppp, endpoint=False)) for i in range(num_coils)]
    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = coil_data[0, 6*ic + 1]
        coils[ic].coefficients[1][0] = coil_data[0, 6*ic + 3]
        coils[ic].coefficients[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, Nt):
            coils[ic].coefficients[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            coils[ic].coefficients[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            coils[ic].coefficients[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            coils[ic].coefficients[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            coils[ic].coefficients[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            coils[ic].coefficients[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].update()

    numpoints = Nt*ppp+1 if ((Nt*ppp) % 2 == 0) else Nt*ppp
    ma = StelleratorSymmetricCylindricalFourierCurve(Nt, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False))
    ma.coefficients[0][0] = 1.
    ma.coefficients[1][0] = 0.01

    ma.update()
    return (coils, ma)

def get_ncsx_data(Nt=25, ppp=10):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    coil_data = np.loadtxt(os.path.join(dir_path, "data", "NCSX_coil_coeffs.dat"), delimiter=',')
    nfp = 3
    num_coils = 3
    coils = [CartesianFourierCurve(Nt, np.linspace(0, 1, Nt*ppp, endpoint=False)) for i in range(num_coils)]
    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = coil_data[0, 6*ic + 1]
        coils[ic].coefficients[1][0] = coil_data[0, 6*ic + 3]
        coils[ic].coefficients[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, Nt):
            coils[ic].coefficients[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            coils[ic].coefficients[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            coils[ic].coefficients[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            coils[ic].coefficients[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            coils[ic].coefficients[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            coils[ic].coefficients[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].update()

    numpoints = Nt*ppp+1 if ((Nt*ppp) % 2 == 0) else Nt*ppp
    ma = StelleratorSymmetricCylindricalFourierCurve(Nt, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False))
    cR = [1.471415400740515, 0.1205306261840785, 0.008016125223436036, -0.000508473952304439, -0.0003025251710853062, -0.0001587936004797397, 3.223984137937924e-06, 3.524618949869718e-05, 2.539719080181871e-06, -9.172247073731266e-06, -5.9091166854661e-06, -2.161311017656597e-06, -5.160802127332585e-07, -4.640848016990162e-08, 2.649427979914062e-08, 1.501510332041489e-08, 3.537451979994735e-09, 3.086168230692632e-10, 2.188407398004411e-11, 5.175282424829675e-11, 1.280947310028369e-11, -1.726293760717645e-11, -1.696747733634374e-11, -7.139212832019126e-12, -1.057727690156884e-12, 5.253991686160475e-13]
    sZ = [0.06191774986623827, 0.003997436991295509, -0.0001973128955021696, -0.0001892615088404824, -2.754694372995494e-05, -1.106933185883972e-05, 9.313743937823742e-06, 9.402864564707521e-06, 2.353424962024579e-06, -1.910411249403388e-07, -3.699572817752344e-07, -1.691375323357308e-07, -5.082041581362814e-08, -8.14564855367364e-09, 1.410153957667715e-09, 1.23357552926813e-09, 2.484591855376312e-10, -3.803223187770488e-11, -2.909708414424068e-11, -2.009192074867161e-12, 1.775324360447656e-12, -7.152058893039603e-13, -1.311461207101523e-12, -6.141224681566193e-13, -6.897549209312209e-14]

    for i in range(Nt):
        ma.coefficients[0][i] = cR[i]
        ma.coefficients[1][i] = sZ[i]
    ma.coefficients[0][Nt] = cR[Nt]

    ma.update()
    currents = [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]
    return (coils, ma, currents)


def get_16_coil_data(Nt=10, ppp=10, at_optimum=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if at_optimum:
        coil_data = np.loadtxt(os.path.join(dir_path, "data", "16_coil_optimal.dat"), delimiter=',')
    else:
        coil_data = np.loadtxt(os.path.join(dir_path, "data", "16_coil_initial.dat"), delimiter=',')
    nfp = 2
    num_coils = 4
    coils = [CartesianFourierCurve(Nt, np.linspace(0, 1, Nt*ppp, endpoint=False)) for i in range(num_coils)]
    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = coil_data[0, 6*ic + 1]
        coils[ic].coefficients[1][0] = coil_data[0, 6*ic + 3]
        coils[ic].coefficients[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, Nt):
            coils[ic].coefficients[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            coils[ic].coefficients[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            coils[ic].coefficients[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            coils[ic].coefficients[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            coils[ic].coefficients[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            coils[ic].coefficients[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].update()

    numpoints = Nt*ppp+1 if ((Nt*ppp) % 2 == 0) else Nt*ppp
    ma = StelleratorSymmetricCylindricalFourierCurve(Nt, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False))
    if at_optimum:
        cR = [0.9887541046929726, -0.06417041409876031, 0.003406984842399879, -0.0002514580242225735, 1.658763311687643e-05, 9.556657435185027e-07, 1.707789451832579e-06, -2.400300429615985e-06, 6.294898825757447e-07, -2.687657374685065e-08]
        sZ = [0.08362053602444068, -0.004570833875474779, 0.0004718058320629744, -6.189245571232168e-05, 3.502857537408366e-06, 2.261126291442485e-06, -2.724550974886944e-06, 6.148016216008801e-07, -8.573492987696066e-08]
        currents = [-317564.5285096774, -310891.1049950164, -303739.9179542562, -317804.448080858]
    else:
        cR = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sZ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        currents = [0., 0., 0., 0.]

    for i in range(Nt):
        ma.coefficients[0][i] = cR[i]
        ma.coefficients[1][i] = sZ[i]
    ma.coefficients[0][Nt] = cR[Nt]
    ma.update()
    return (coils, ma, currents)
