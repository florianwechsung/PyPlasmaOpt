import pytest
import numpy as np
from pyplasmaopt import  BiotSavart, get_24_coil_data, CoilCollection, \
    QuasiSymmetricField
from pyplasmaopt.qfm_surface import QfmSurface
from pyplasmaopt.finite_differences import finite_difference_derivative

def test_params_full():
    """
    Check that output has correct shape
    Check that parameter derivatives match finite differences
    """
    mmax = 1
    nmax = 1
    nfp = 2
    ntheta = 3
    nphi = 3
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    volume = 1
    qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)
    
    params = np.random.rand((2*qfm.mnmax-1))
    params_full = qfm.params_full(params)
    assert(np.ndim(params)==np.ndim(params_full))
    assert(len(params)+1==len(params_full))
    
    d_params_full = qfm.d_params_full(params)
    d_params_full_fd = finite_difference_derivative(params, qfm.params_full, epsilon=1e-5, 
                                          method='centered')
    
    error = np.abs(d_params_full[:,0:len(params)].T - d_params_full_fd)
    assert(np.allclose(d_params_full[:,0:len(params)].T,d_params_full_fd))    
    
def test_position():
    """
    Check that output has correct shape
    Check that parameter derivatives match finite differences
    """
    mmax = 3
    nmax = 2
    nfp = 2
    ntheta = mmax*3
    nphi = nmax*3
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    volume = 1
    qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)
    
    params = np.random.rand((2*qfm.mnmax-1))
    R, Z = qfm.position(params)
    assert(np.all(np.shape(R)==np.shape(qfm.thetas)))
    assert(np.all(np.shape(Z)==np.shape(qfm.thetas)))
    d_R, d_Z = qfm.d_position(params)
    
    R_fun = lambda params : qfm.position(params)[0]
    Z_fun = lambda params : qfm.position(params)[1]
    d_R_fd = finite_difference_derivative(params, R_fun, epsilon=1e-3, 
                                          method='centered')
    d_Z_fd = finite_difference_derivative(params, Z_fun, epsilon=1e-3, 
                                          method='centered')
    assert(np.allclose(d_R_fd,d_R))
    assert(np.allclose(d_Z_fd,d_Z))
    
def test_position_derivatives():
    """
    Check that output has correct shape
    Check that perfect derivatives integrate to zero
    Check that parameter derivatives match finite differences
    """
    mmax = 3
    nmax = 3
    nfp = 2
    ntheta = mmax*3
    nphi = nmax*3
    volume = 1
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)
    
    params = np.random.rand((2*qfm.mnmax-1))
    dRdtheta, dRdphi, dZdtheta, dZdphi = qfm.position_derivatives(params)
    assert(np.all(np.shape(dRdtheta)==np.shape(qfm.thetas)))
    assert(np.all(np.shape(dRdphi)==np.shape(qfm.thetas)))
    assert(np.all(np.shape(dZdtheta)==np.shape(qfm.thetas)))
    assert(np.all(np.shape(dZdphi)==np.shape(qfm.thetas)))
    
    assert(np.allclose(np.sum(dRdtheta,axis=1),0))
    assert(np.allclose(np.sum(dRdphi,axis=0),0))
    assert(np.allclose(np.sum(dZdtheta,axis=1),0))
    assert(np.allclose(np.sum(dZdphi,axis=0),0))

    d_dRdtheta, d_dRdphi, d_dZdtheta, d_dZdphi = qfm.d_position_derivatives(params)
    
    dRdtheta_fun = lambda params : qfm.position_derivatives(params)[0]
    dRdphi_fun = lambda params : qfm.position_derivatives(params)[1]
    dZdtheta_fun = lambda params : qfm.position_derivatives(params)[2]
    dZdphi_fun = lambda params : qfm.position_derivatives(params)[3]

    d_dRdtheta_fd = finite_difference_derivative(params, dRdtheta_fun, 
                                          epsilon=1e-3, method='centered')
    d_dRdphi_fd = finite_difference_derivative(params, dRdphi_fun, 
                                          epsilon=1e-3, method='centered')
    d_dZdtheta_fd = finite_difference_derivative(params, dZdtheta_fun, 
                                          epsilon=1e-3, method='centered')
    d_dZdphi_fd = finite_difference_derivative(params, dZdphi_fun, 
                                          epsilon=1e-3, method='centered')

    assert(np.allclose(d_dRdtheta_fd,d_dRdtheta))
    assert(np.allclose(d_dRdphi_fd,d_dRdphi))

    assert(np.allclose(d_dZdtheta_fd,d_dZdtheta))
    assert(np.allclose(d_dZdphi_fd,d_dZdphi))
    
def test_unit_normal():
    """
    Check output has correct shape
    Check that parameter derivatives match finite differences
    """
    mmax = 3
    nmax = 3
    nfp = 2
    ntheta = mmax*3
    nphi = nmax*3
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    volume = 1
    qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)
        
    params = 0.1*np.ones((2*qfm.mnmax-1))
    
    nR, nP, nZ = qfm.normal(params)
    
    assert(np.all(np.shape(nR)==np.shape(qfm.thetas)))
    assert(np.all(np.shape(nP)==np.shape(qfm.thetas)))
    assert(np.all(np.shape(nZ)==np.shape(qfm.thetas)))

    d_nR, d_nP, d_nZ = qfm.d_normal(params)
    
    nR_fun = lambda params: qfm.normal(params)[0]
    nP_fun = lambda params: qfm.normal(params)[1]
    nZ_fun = lambda params: qfm.normal(params)[2]

    d_nR_fd = finite_difference_derivative(params, nR_fun, 
                                          epsilon=1e-6, method='centered')
    d_nP_fd = finite_difference_derivative(params, nP_fun, 
                                          epsilon=1e-6, method='centered')
    d_nZ_fd = finite_difference_derivative(params, nZ_fun, 
                                          epsilon=1e-6, method='centered')

    assert(np.allclose(d_nR,d_nR_fd,rtol=1e-4))    
    assert(np.allclose(d_nP,d_nP_fd,rtol=1e-4))    
    assert(np.allclose(d_nZ,d_nZ_fd,rtol=1e-4))    
    
def test_norm_normal():
    """
    Check output has correct shape
    Check that parameter derivatives match finite differences
    """
    mmax = 3
    nmax = 3
    nfp = 2
    ntheta = mmax*3
    nphi = nmax*3
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    volume = 1
    qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)
    
    params = np.random.rand((2*qfm.mnmax-1))
        
    N = qfm.norm_normal(params)
    
    assert(np.all(np.shape(N)==np.shape(qfm.thetas)))
    
    d_N = qfm.d_norm_normal(params)

    d_N_fd = finite_difference_derivative(params, qfm.norm_normal, 
                                          epsilon=1e-4, method='centered')
    
    assert(np.allclose(d_N_fd,d_N))    
    
def test_B_from_points():
    """
    Check that parameter derivatives match finite differences
    """
    mmax = 3
    nmax = 3
    nfp = 2
    ntheta = mmax*3
    nphi = nmax*3
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    volume = 1
    qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)
    
    params = np.random.rand((2*qfm.mnmax-1))
    
    d_B = qfm.d_B_from_points(params)

    d_B_fd = finite_difference_derivative(params, qfm.B_from_points, 
                                          epsilon=1e-5, method='centered')
    
    assert(np.allclose(d_B_fd,d_B))    
    
def test_quadratic_flux():
    """
    Check that parameter derivatives match finite differences
    """
    mmax = 3
    nmax = 3
    nfp = 2
    ntheta = mmax*3
    nphi = nmax*3
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    volume = 1
    qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)
    
    params = np.random.rand((2*qfm.mnmax-1))

    d_quadratic_flux = qfm.d_quadratic_flux(params)
    d_quadratic_flux_fd = finite_difference_derivative(params, qfm.quadratic_flux, 
                                          epsilon=1e-5, method='centered')
    
    assert(np.allclose(d_quadratic_flux_fd,d_quadratic_flux))    
    
def test_ft_surface():
    """
    Check that IFT matches initial surfaces
    """
    mmax = 3
    nmax = 3
    nfp = 2
    ntheta = 100
    nphi = 100
    (coils, _, ma, _) = get_24_coil_data(nfp=nfp, ppp=20)
    currents = len(coils) * [1e4]
    stellerator = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(stellerator.coils, stellerator.currents)
    volume = 1
    qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)
    
    params = np.random.rand((2*qfm.mnmax-1))

    R, Z = qfm.position(params)
    Rbc, Zbs = qfm.ft_surface(params,mmax,nmax)
    
    mnmax, xm, xn = qfm.init_modes(mmax,nmax)
    xn = xn*qfm.nfp
    
    nax = np.newaxis
    xm = xm[:,nax,nax]
    xn = xn[:,nax,nax]
    thetas = qfm.thetas[nax,...]
    phis = qfm.phis[nax,...]
    angle = xm*thetas - xn*phis
    
    R_ift = np.sum(Rbc[:,nax,nax]*np.cos(angle),axis=0)
    Z_ift = np.sum(Zbs[:,nax,nax]*np.sin(angle),axis=0)

    assert(np.allclose(R_ift,R))
    assert(np.allclose(Z_ift,Z))