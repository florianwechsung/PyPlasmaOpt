import pytest
import netCDF4 as nc
from pyplasmaopt import CoilCollection, get_ncsx_data
import numpy as np
import cppplasmaopt as cpp
import os


def test_magnetic_field_in_ncsx_is_correct():
    nfp = 3
    (coils, ma, currents) = get_ncsx_data(Nt=25, ppp=20)
    stellarator = CoilCollection(coils, currents, nfp, True)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path, "..", "pyplasmaopt", "data", "ncsx", "mgrid_c09r00_modularOnly.nc")
    ds = nc.Dataset(filepath)
    Br = ds['br_001'][:,:,:]+ds['br_002'][:,:,:]+ds['br_003'][:,:,:]
    Bp = ds['bp_001'][:,:,:]+ds['bp_002'][:,:,:]+ds['bp_003'][:,:,:]
    Bz = ds['bz_001'][:,:,:]+ds['bz_002'][:,:,:]+ds['bz_003'][:,:,:]
    Rs = np.linspace(0.436, 2.436, 5, endpoint=True)
    phis = np.linspace(0, 2*np.pi/3, 4, endpoint=False)
    Zs = np.linspace(-1, 1, 9, endpoint=True)
    gammas          = [coil.gamma for coil in stellarator.coils]
    dgamma_by_dphis = [coil.dgamma_by_dphi[:, 0, :] for coil in stellarator.coils]
    currents = stellarator.currents
    avg_rel_err = 0
    max_rel_err = 0
    for i, phi in enumerate(phis):
        for j, Z in enumerate(Zs):
            for k, R in enumerate(Rs):
                x = R * np.cos(phi)
                y = R * np.sin(phi)
                Bxyz = np.zeros((1, 3))
                xyz = np.asarray([[x, y, Z]])
                cpp.biot_savart_B_only(xyz, gammas, dgamma_by_dphis, currents, Bxyz)
                trueBx = np.cos(phi) * Br[i, j, k] - np.sin(phi) * Bp[i, j, k]
                trueBy = np.sin(phi) * Br[i, j, k] + np.cos(phi) * Bp[i, j, k]
                trueBz = Bz[i, j, k]
                trueB = np.asarray([[trueBx, trueBy, trueBz]])
                err = np.linalg.norm(Bxyz-trueB)/np.linalg.norm(trueB)
                avg_rel_err += err
                max_rel_err = max(max_rel_err, err)
                assert err < 2e-2
    print('avg_rel_err', avg_rel_err/((len(phis)*len(Zs)*len(Rs))))
    print('max_rel_err', max_rel_err)
