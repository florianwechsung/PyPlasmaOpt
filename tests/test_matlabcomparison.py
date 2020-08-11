import pytest
import numpy as np
from pyplasmaopt import BiotSavart, SquaredMagneticFieldNormOnCurve, SquaredMagneticFieldGradientNormOnCurve, get_24_coil_data, BiotSavartQuasiSymmetricFieldDifference, QuasiSymmetricField, CoilCollection
from math import pi
import os 


"""
Obtained these results with plasmopt@14c162691fc71cd80febd20b935ad1f5a336d9bd but

diff --git a/scripts/problem2/coilDataType.m b/scripts/problem2/coilDataType.m
index 01789b4..64cd0c6 100644
--- a/scripts/problem2/coilDataType.m
+++ b/scripts/problem2/coilDataType.m
@@ -35,7 +35,7 @@
                 obj.coil_coeffs = obj.coil_coeffs(1:obj.Nt, :);
             end

-            obj.PPP = 10;% number of points per period
+            obj.PPP = 20;% number of points per period
             obj.M = obj.PPP * obj.Nt;% number of discretization points for each coil

index 9cffb47..22d03d1 100644
--- a/scripts/problem2/driver.m
+++ b/scripts/problem2/driver.m
@@ -28,7 +28,7 @@
     dof = 1 + 2*magnetic_axis_data.top + numel(coilData.I)  + numel(coilData.coil_coeffs);
     x = zeros(1,dof);

-
+    coilData.I = coilData.I + 1e4

"""


def test_biot_savart_same_results_as_matlab():
    num_coils = 6
    nfp = 2
    coils, currents, ma, eta_bar = get_24_coil_data(nfp=nfp, ppp=20)
    currents = num_coils * [1e4]
    coil_collection = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(coil_collection.coils, coil_collection.currents)
    points = np.asarray([
        [1.079860105000000,                0., 0.],
        [1.078778093231020, 0.041861502907184, -0.006392709264512],
    ])
    matlab_res = np.asarray([
        [0,                 -0.044495549447737, 0.005009283509639],
        [0.002147564148695, -0.044454924339257, 0.004992777089330],
    ])
    bs.compute(points, use_cpp=True)
    print(bs.B, "\n", matlab_res)
    assert np.allclose(bs.B, matlab_res)

    J = SquaredMagneticFieldNormOnCurve(ma, bs)
    J0 = J.J()
    assert abs(0.5 * J0 - 0.007179654002556) < 1e-10

    J = SquaredMagneticFieldGradientNormOnCurve(ma, bs)
    J0 = J.J()
    assert abs(0.5 * J0 - 0.014329772542444) < 1e-10


    if __name__ == "__main__":
        ax = None
        for i in range(0, len(coils)):
            ax = coils[i].plot(ax=ax, show=False)
        ma.plot(ax=ax)

def test_quasi_symmetric_difference_same_results_as_matlab():
    num_coils = 6
    nfp = 2
    coils, currents, ma, eta_bar = get_24_coil_data(nfp=nfp, ppp=20)
    currents = num_coils * [1e4]
    coil_collection = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(coil_collection.coils, coil_collection.currents)
    bs.set_points(ma.gamma)
    qsf = QuasiSymmetricField(-2.25, ma)
    J = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
    print(0.5 * J.J_L2())
    print(0.5 * J.J_H1())
    assert abs(3.486875802492926 - 0.5 * J.J_L2()) < 1e-5
    assert abs(8.296004257157044 - 0.5 * J.J_H1()) < 1e-5

def test_sigma():
    nfp = 2
    _, _, ma, _ = get_24_coil_data(nfp=nfp, ppp=40)
    eta_bar = -2.25
    qsf = QuasiSymmetricField(eta_bar, ma)
    sigma = qsf.sigma
    sigma_matlab = np.asarray([
        -0.000000000000000, -0.056185092625620, -0.112382986380741, -0.168606294739314, -0.224867254805832, -0.281177536480886, -0.337548048436650, -0.393988739822343, -0.450508396607183, -0.507114431454616, -0.563812666008344, -0.620607104459419, -0.677499697256753, -0.734490093823280, -0.791575383149675, -0.848749821160578, -0.906004543788638, -0.963327264754031, -1.020701957136678, -1.078108517951100, -1.135522415096348, -1.192914316262969, -1.250249699643717, -1.307488446623194, -1.364584417023181, -1.421485007964753, -1.478130697985200, -1.534454578727070, -1.590381877307561, -1.645829473387029, -1.700705415991899, -1.754908446313264, -1.808327533997705, -1.860841435865964, -1.912318287526287, -1.962615239972233, -2.011578154939745, -2.059041374503194, -2.104827582059986, -2.148747773417560, -2.190601358069108, -2.230176411822216, -2.267250102609442, -2.301589311429245, -2.332951469797540, -2.361085633687722, -2.385733811557786, -2.406632560578662, -2.423514860486942, -2.436112268528834, -2.444157351739409, -2.447386384385629, -2.445542288955833, -2.438377788865210, -2.425658730435253, -2.407167521167246, -2.382706621428404, -2.352102018033593, -2.315206601494947, -2.271903364568463, -2.222108338719867, -2.165773187697747, -2.102887383784994, -2.033479902503788, -1.957620385304120, -1.875419736514573, -1.787030139750048, -1.692644499001194, -1.592495329566807, -1.486853142563567, -1.376024382731296, -1.260348991572805, -1.140197675702519, -1.015968963133882, -0.888086127997064, -0.756994057124279, -0.623156120721537, -0.487051094933613, -0.349170167713112, -0.210014042374294, -0.070090136929009, 0.070090136929009, 0.210014042374294, 0.349170167713112, 0.487051094933614, 0.623156120721537, 0.756994057124279, 0.888086127997064, 1.015968963133882, 1.140197675702519, 1.260348991572805, 1.376024382731296, 1.486853142563567, 1.592495329566807, 1.692644499001194, 1.787030139750048, 1.875419736514573, 1.957620385304121, 2.033479902503789, 2.102887383784994, 2.165773187697748, 2.222108338719868, 2.271903364568463, 2.315206601494947, 2.352102018033593, 2.382706621428404, 2.407167521167246, 2.425658730435254, 2.438377788865211, 2.445542288955834, 2.447386384385629, 2.444157351739409, 2.436112268528835, 2.423514860486943, 2.406632560578662, 2.385733811557786, 2.361085633687722, 2.332951469797541, 2.301589311429245, 2.267250102609442, 2.230176411822216, 2.190601358069108, 2.148747773417560, 2.104827582059986, 2.059041374503194, 2.011578154939746, 1.962615239972234, 1.912318287526287, 1.860841435865964, 1.808327533997705, 1.754908446313264, 1.700705415991898, 1.645829473387029, 1.590381877307561, 1.534454578727070, 1.478130697985200, 1.421485007964753, 1.364584417023181, 1.307488446623194, 1.250249699643717, 1.192914316262969, 1.135522415096348, 1.078108517951100, 1.020701957136678, 0.963327264754031, 0.906004543788638, 0.848749821160577, 0.791575383149675, 0.734490093823280, 0.677499697256752, 0.620607104459418, 0.563812666008344, 0.507114431454616, 0.450508396607183, 0.393988739822343, 0.337548048436650, 0.281177536480886, 0.224867254805832, 0.168606294739314, 0.112382986380741, 0.056185092625620,
   ])
    assert all(np.abs(sigma-sigma_matlab) < 1e-8)

def test_iota():
    nfp = 2
    _, _, ma, _ = get_24_coil_data(nfp=nfp, ppp=20)
    eta_bar = -2.25
    qsf = QuasiSymmetricField(eta_bar, ma)
    iota = qsf.iota
    assert (abs(iota-0.038138935935663) < 1e-6)

def test_Bqs():
    nfp = 2
    _, _, ma, _ = get_24_coil_data(nfp=nfp, ppp=100)
    eta_bar = -2.25
    qsf = QuasiSymmetricField(eta_bar, ma)
    iota = qsf.iota
    assert (abs(iota-0.038138935935663) < 1e-6)

    Bqs = qsf.B
    matlab_Bqs = np.asarray([
        [0                ,   0.988522982100515,  -0.151070559206963],
        [-0.010321587904285,   0.988472377058748,  -0.151049080152652],
        [-0.020641457945632,   0.988320578381101,  -0.150984650088425]])
    assert np.all(np.abs(Bqs[:3,:] - matlab_Bqs) < 1e-6)

    dBqs_by_dX = qsf.dB_by_dX
    assert np.allclose(dBqs_by_dX[0, : ,:].T, dBqs_by_dX[0, :, :])
    matlab_dBqs_by_dx = np.asarray([
        [ 0.000000000000000,  -1.125930207500723 ,  0.616014274297893],
        [ 0.030176925011424,  -1.125619099754597 ,  0.615869343228699],
        [ 0.060336364146126,  -1.124686016345258 ,  0.615434630935356]])
    assert np.all(np.abs(matlab_dBqs_by_dx-dBqs_by_dX[:3,0,:])<2e-6)
    
    matlab_dBqs_by_dy = np.asarray([
        [-1.125930207500723 , -0.000000000000000 , -0.000000000000000],
        [-1.125619099754597 , -0.022146228771976 ,  0.010281638036739],
        [-1.124686016345258 , -0.044277063777819 ,  0.020556281775495]])
    assert np.all(np.abs(matlab_dBqs_by_dy-dBqs_by_dX[:3,1,:])<2e-6)
    matlab_dBqs_by_dz = np.asarray([
       [0.616014274297891,  -0.000000000000000,  -0.000000000000000],
       [0.615869343228698,   0.010281638036739,  -0.008030696239448],
       [0.615434630935353,   0.020556281775495,  -0.016059300368306]])
    assert np.all(np.abs(matlab_dBqs_by_dz-dBqs_by_dX[:3,2,:])<2e-6)
