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

    if at_optimum:
        currents = [1e5 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]
        eta_bar = -2.105800979374183
    else:
        currents = 6 * [0.]
        eta_bar = -2.25

    return (coils, currents, ma, eta_bar)

def make_flat_stell(Nt_coils=6, Nt_ma=6, nfp=3, ppp=20, num_coils=3, major_radius=1.4, minor_radius=0.33, copies=1, kick=False, magnitude=0.2): #kwargs are based on NCSX specs

    coils = [CartesianFourierCurve(Nt_coils, np.linspace(0, 1, Nt_coils*ppp, endpoint=False)) for i in range(num_coils)]

    field_period_angle = 2*np.pi/nfp #The angle occupied by each field period.
    total_coils = 2*nfp*num_coils #Total number of coils in device assuming stellarator symmetry.
    shift = np.pi/total_coils #Half the angle between each coil in the device. This is the angle between the first coil in a field period and the \
            # beginning of the field period itself, as well as the angle between the last field period and the end of the field period itself.
    phi_vals = np.linspace(shift, field_period_angle-shift, 2*num_coils, endpoint=True) #This gets the proper angles of each coil in the field period.
    phi_vals = phi_vals[:len(phi_vals)//2] #Due to stellarator symmetry, we only need the first half of the list - the other coils are generated \
            # using stellaratory symmetry downstream.
    assert len(coils)==len(phi_vals) #Sanity check.

    #These Fourier coefficients come from expressing the coils in cylindrical coordinates.
    X0 = major_radius*np.cos(phi_vals)
    Y0 = major_radius*np.sin(phi_vals)
    Z0 = np.repeat(0,len(phi_vals))
    X1 = minor_radius*np.cos(phi_vals)
    Y1 = minor_radius*np.sin(phi_vals)
    Z1 = np.repeat(minor_radius,len(phi_vals))
    if kick:
        Z0 = 4*magnitude*(minor_radius)*np.sin(1*nfp*phi_vals)
        X0 = (major_radius + (minor_radius*magnitude)*np.cos(1*nfp*phi_vals))*np.cos(phi_vals)
        Y0 = (major_radius + (minor_radius*magnitude)*np.cos(1*nfp*phi_vals))*np.sin(phi_vals)

    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = X0[ic]
        coils[ic].coefficients[1][0] = Y0[ic]
        coils[ic].coefficients[2][0] = Z0[ic]
        coils[ic].coefficients[0][1] = 0
        coils[ic].coefficients[0][2] = X1[ic]
        coils[ic].coefficients[1][1] = 0
        coils[ic].coefficients[1][2] = Y1[ic]
        coils[ic].coefficients[2][1] = Z1[ic]
        coils[ic].coefficients[2][2] = 0
        for io in range(2, Nt_coils):
            coils[ic].coefficients[0][2*io-1] = 0
            coils[ic].coefficients[0][2*io] = 0
            coils[ic].coefficients[1][2*io-1] = 0
            coils[ic].coefficients[1][2*io] = 0
            coils[ic].coefficients[2][2*io-1] = 0
            coils[ic].coefficients[2][2*io] = 0
        coils[ic].update()

    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    if copies == 0:
        mas = StelleratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False))
        mas.coefficients[0][0] = major_radius
        mas.coefficients[1][0] = 0
        if kick:
            mas.coefficients[0][1] =   minor_radius*magnitude
            mas.coefficients[1][0] =   4*minor_radius*magnitude
        mas.update()
    else:
        mas = [StelleratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False)) for i in range(copies)]    
        for j in range(copies):
            mas[j].coefficients[0][0] = major_radius
            mas[j].coefficients[1][0] = 0
            if kick:
                mas[j].coefficients[0][1] =   minor_radius*magnitude
                mas[j].coefficients[1][0] =   4*minor_radius*magnitude
            mas[j].update()

    #total_current = 7497492.944369065 #From NCSX
    mu_nought = 4*np.pi*1e-7 #SI units
    coil_current = 2*np.pi*major_radius/mu_nought/total_coils #From Ampere's Law
    currents = [-1*coil_current]*num_coils #Normalized to give B=1 on the axis. The negative is for a sign convention used in the rest of the program.
#     if kick:
#         currents[0] += currents[0]*magnitude #Perturbation so the solver doesn't get stuck.

    return coils, mas, currents

def reload_stell(sourcedir,Nt_coils=25,Nt_ma=25,ppp=10,nfp=3,stellID=0,num_coils=3,copies=1):
    "Data for coils, currents, and the magnetic axis is pulled from sourcedir. \
            There is only need to input *unique* coils - the others will be created using CoilCollection as usual. \
            The function parameters have the same meaning as in get_ncsx_data. \
            Note that Nt_coils, Nt_ma, ppp, and nfp MUST be the same as in the original stellarator."

    coil_data = np.loadtxt(os.path.join(sourcedir,'coilCoeffs.txt'))

    repeat_factor = len(coil_data)/num_coils #How many consecutive lines of coil_data belong to each coil.

    shaped_coil_data = [] #List. Indices: shaped_coil_data[unique coil index][coefficients sublist index][coefficient index]
    for vecind,vec in enumerate(coil_data):
        if vecind%repeat_factor==0:
            intermed = []
            intermed.append(vec.tolist())
        else:
            intermed.append(vec.tolist())
        if len(intermed)==repeat_factor:
            shaped_coil_data.append(intermed)

    coils = [CartesianFourierCurve(Nt_coils, np.linspace(0, 1, Nt_coils*ppp, endpoint=False)) for i in range(num_coils)] #Create blank coils object of the proper class.
    for coilind in range(num_coils):
        for sublistind in range(len(coils[coilind].coefficients)):
            for coeffind in range(len(coils[coilind].coefficients[sublistind])):
                coils[coilind].coefficients[sublistind][coeffind] = shaped_coil_data[coilind][sublistind][coeffind]
        coils[coilind].update()

    ma_raw = []
    with open(os.path.join(sourcedir,'maCoeffs_%d.txt'%stellID),'r') as f:
        for line in f:
            linelist = [float(coeff) for coeff in line.strip().split()]
            ma_raw.append(linelist)

    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    mas = [StelleratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False)) for i in range(copies)]

    for j in range(copies):
        for ind1 in range(len(ma_raw)):
            for ind2 in range(len(ma_raw[ind1])):
                    mas[j].coefficients[ind1][ind2] = ma_raw[ind1][ind2]
        mas[j].update() #The magnetic axes should now be ready to go.

    currents = np.loadtxt(os.path.join(sourcedir,'currents_%d.txt'%stellID)).tolist() #Only the currents for the three unique coils need to be imported.

    eta_bar = np.loadtxt(os.path.join(sourcedir,'eta_bar_%d.txt'%stellID)) #Reload eta_bar from previous run as a starting point.

    return (coils, mas, currents, eta_bar)

def get_ncsx_data(Nt_coils=25, Nt_ma=25, ppp=10, copies=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    coil_data = np.loadtxt(os.path.join(dir_path, "data", "NCSX_coil_coeffs.dat"), delimiter=',')
    nfp = 3
    num_coils = 3
    coils = [CartesianFourierCurve(Nt_coils, np.linspace(0, 1, Nt_coils*ppp, endpoint=False)) for i in range(num_coils)]
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

    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    mas = [StelleratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False)) for i in range(copies)]
    cR = [1.471415400740515, 0.1205306261840785, 0.008016125223436036, -0.000508473952304439, -0.0003025251710853062, -0.0001587936004797397, 3.223984137937924e-06, 3.524618949869718e-05, 2.539719080181871e-06, -9.172247073731266e-06, -5.9091166854661e-06, -2.161311017656597e-06, -5.160802127332585e-07, -4.640848016990162e-08, 2.649427979914062e-08, 1.501510332041489e-08, 3.537451979994735e-09, 3.086168230692632e-10, 2.188407398004411e-11, 5.175282424829675e-11, 1.280947310028369e-11, -1.726293760717645e-11, -1.696747733634374e-11, -7.139212832019126e-12, -1.057727690156884e-12, 5.253991686160475e-13]
    sZ = [0.06191774986623827, 0.003997436991295509, -0.0001973128955021696, -0.0001892615088404824, -2.754694372995494e-05, -1.106933185883972e-05, 9.313743937823742e-06, 9.402864564707521e-06, 2.353424962024579e-06, -1.910411249403388e-07, -3.699572817752344e-07, -1.691375323357308e-07, -5.082041581362814e-08, -8.14564855367364e-09, 1.410153957667715e-09, 1.23357552926813e-09, 2.484591855376312e-10, -3.803223187770488e-11, -2.909708414424068e-11, -2.009192074867161e-12, 1.775324360447656e-12, -7.152058893039603e-13, -1.311461207101523e-12, -6.141224681566193e-13, -6.897549209312209e-14]
    for j in range(copies):
        for i in range(Nt_ma):
            mas[j].coefficients[0][i] = cR[i]
            mas[j].coefficients[1][i] = sZ[i]
        mas[j].coefficients[0][Nt_ma] = cR[Nt_ma]
        mas[j].update()

    currents = [c/1.474 for c in [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]] # normalise to get a magnetic field of around 1 at the axis
    return (coils, mas, currents)


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
