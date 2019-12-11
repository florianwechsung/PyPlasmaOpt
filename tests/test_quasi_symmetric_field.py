import numpy as np
import pytest
from pyplasmaopt import QuasiSymmetricField, get_matt_data

def test_sigma():
    nfp = 2
    _, ma = get_matt_data(Nt=4, nfp=nfp, ppp=20)
    eta_bar = -2.25
    B_qs = QuasiSymmetricField(eta_bar, ma)
    sigma, iota = B_qs.solve_state()
    sigma_matlab = np.asarray([0.0000, -0.1117, -0.2235, -0.3355, -0.4477, -0.5603, -0.6733, -0.7866, -0.9003, -1.0143, -1.1284, -1.2425, -1.3561, -1.4691, -1.5807, -1.6906, -1.7978, -1.9016, -2.0008, -2.0942, -2.1805, -2.2579, -2.3247, -2.3791, -2.4189, -2.4422, -2.4467, -2.4305, -2.3918, -2.3291, -2.2413, -2.1277, -1.9884, -1.8238, -1.6352, -1.4245, -1.1941, -0.9469, -0.6863, -0.4157, -0.1393, 0.1393, 0.4157, 0.6863, 0.9469, 1.1941, 1.4245, 1.6352, 1.8238, 1.9884, 2.1277, 2.2413, 2.3291, 2.3918, 2.4305, 2.4467, 2.4422, 2.4189, 2.3791, 2.3247, 2.2579, 2.1805, 2.0942, 2.0008, 1.9016, 1.7978, 1.6906, 1.5807, 1.4691, 1.3561, 1.2425, 1.1284, 1.0143, 0.9003, 0.7866, 0.6733, 0.5603, 0.4477, 0.3355, 0.2235, 0.1117])
    assert all(np.abs(sigma-sigma_matlab) < 1e-2)

def test_iota():
    nfp = 2
    _, ma = get_matt_data(Nt=4, nfp=nfp, ppp=100)
    eta_bar = -2.25
    B_qs = QuasiSymmetricField(eta_bar, ma)
    sigma, iota = B_qs.solve_state()
    assert (abs(iota-0.038138935935663) < 1e-6)

def test_Bqs():
    nfp = 2
    _, ma = get_matt_data(Nt=4, nfp=nfp, ppp=100)
    eta_bar = -2.25
    B_qs = QuasiSymmetricField(eta_bar, ma)
    sigma, iota = B_qs.solve_state()
    assert (abs(iota-0.038138935935663) < 1e-6)

    Bqs = B_qs.B()
    matlab_Bqs = np.asarray([
        [0                ,   0.988522982100515,  -0.151070559206963],
        [-0.010321587904285,   0.988472377058748,  -0.151049080152652],
        [-0.020641457945632,   0.988320578381101,  -0.150984650088425]])
    assert np.all(np.abs(Bqs[:3,:] - matlab_Bqs) < 1e-6)

    dBqs_by_dX = B_qs.dB_by_dX()
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
