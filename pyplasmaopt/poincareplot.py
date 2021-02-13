import numpy as np
import cppplasmaopt as cpp
from .logging import info_all, info_all_sync
from math import sqrt, copysign
def sign(y):
    return copysign(1, y)

def compute_field_lines(biotsavart, nperiods=200, batch_size=8, magnetic_axis_radius=1, max_thickness=0.5, delta=0.01, steps_per_period=100):

    def cylindrical_to_cartesian(rphiz):
        xyz = np.zeros(rphiz.shape)
        xyz[:, 0] = rphiz[:, 0] * np.cos(rphiz[:, 1])
        xyz[:, 1] = rphiz[:, 0] * np.sin(rphiz[:, 1])
        xyz[:, 2] = rphiz[:, 2]
        return xyz

    gammas                 = [coil.gamma() for coil in biotsavart.coils]
    dgamma_by_dphis        = [coil.gammadash() for coil in biotsavart.coils]
    largest = [0.]

    def rhs(phi, rz):
        nparticles = rz.shape[0]//2
        while phi >= np.pi:
            phi -= 2*np.pi
        rz = rz.reshape((nparticles, 2))
        rphiz = np.zeros((nparticles, 3))
        rphiz[:, 0] = rz[:, 0]
        rphiz[:, 1] = phi
        rphiz[:, 2] = rz[:, 1]
        xyz = cylindrical_to_cartesian(rphiz)

        Bxyz = np.zeros((nparticles, 3))
        cpp.biot_savart_B_only(xyz, gammas, dgamma_by_dphis, biotsavart.coil_currents, Bxyz)

        rhs_xyz = np.zeros((nparticles, 3))
        rhs_xyz[:, 0] = Bxyz[:, 0]
        rhs_xyz[:, 1] = Bxyz[:, 1]
        rhs_xyz[:, 2] = Bxyz[:, 2]

        # Two different ways of deriving the rhs.
        if False:
            rhs_phi = (np.cos(phi) * Bxyz[:, 1] - np.sin(phi)*Bxyz[:, 0])/rphiz[:, 0]
            rhs_rz = np.zeros((nparticles, 2))
            rhs_rz[:, 0] = (rhs_xyz[:, 0] * np.cos(phi) + rhs_xyz[:, 1] * np.sin(phi))/rhs_phi
            rhs_rz[:, 1] = rhs_xyz[:, 2]/rhs_phi
            return rhs_rz.flatten()
        else:
            B_phi = (np.cos(phi) * Bxyz[:, 1] - np.sin(phi)*Bxyz[:, 0])
            B_r = np.cos(phi) * Bxyz[:, 0] + np.sin(phi)*Bxyz[:, 1]
            B_z = Bxyz[:, 2]
            rhs_rz = np.zeros(rz.shape)
            rhs_rz[:, 0] = rphiz[:, 0] * B_r/B_phi
            rhs_rz[:, 1] = rphiz[:, 0] * B_z/B_phi
            return rhs_rz.flatten()


    from scipy.integrate import solve_ivp, RK45, OdeSolution
    from math import pi

    res = []
    nt = int(steps_per_period * nperiods)
    tspan = [0, 2*pi*nperiods]
    t_eval = np.linspace(0, tspan[-1], nt+1)
    i = 0
    while (i+1)*batch_size*delta < max_thickness:
        y0 = np.zeros((batch_size, 2))
        y0[:, 0] = np.linspace(magnetic_axis_radius + i*batch_size*delta, magnetic_axis_radius+(i+1)*batch_size*delta, batch_size, endpoint=False)
        t = tspan[0]
        solver = RK45(rhs, tspan[0], y0.flatten(), tspan[-1], rtol=1e-9, atol=1e-09)
        ts = [0]
        denseoutputs = []
        while t < tspan[-1]:
            solver.step()
            if solver.t < t + 1e-10: # no progress --> abort
                break
            t = solver.t
            ts.append(solver.t)
            denseoutputs.append(solver.dense_output())
        if t >= tspan[1]:
            odesol = OdeSolution(ts, denseoutputs)
            res.append(odesol(t_eval))
            print(y0[0, 0], "to", y0[-1, 0], "-> success")
        else:
            print(y0[0, 0], "to", y0[-1, 0], "-> fail")
        #     break
        i += 1

    nparticles = len(res) * batch_size

    rphiz = np.zeros((nparticles, nt, 3))
    xyz = np.zeros((nparticles, nt, 3))
    phi_no_mod = t_eval.copy()
    for i in range(nt):
        while t_eval[i] >= np.pi:
            t_eval[i] -= 2*np.pi
        rphiz[:, i, 1] = t_eval[i]
    for j in range(len(res)):
        for i in range(nt):
            rz = res[j][:, i].reshape((batch_size, 2))
            rphiz[j*batch_size:(j+1)*batch_size, i, 0] = rz[:, 0]
            rphiz[j*batch_size:(j+1)*batch_size, i, 2] = rz[:, 1]
    for i in range(nt):
        xyz[:, i, :] = cylindrical_to_cartesian(rphiz[:, i, :])


    absB = np.zeros((nparticles, nt))
    tmp = np.zeros((nt, 3))
    for j in range(nparticles):
        tmp[:] = 0
        cpp.biot_savart_B_only(xyz[j, :, :], gammas, dgamma_by_dphis, biotsavart.coil_currents, tmp)
        absB[j, :] = np.linalg.norm(tmp, axis=1)

    return rphiz, xyz, absB, phi_no_mod[:-1]

def find_magnetic_axis(biotsavart, n, rguess, output='cylindrical'):
    assert output in ['cylindrical', 'cartesian']
    from scipy.spatial.distance import cdist
    from scipy.optimize import fsolve
    points = np.linspace(0, 2*np.pi, n, endpoint=False).reshape((n, 1))
    oneton = np.asarray(range(0, n)).reshape((n, 1))
    fak = 2*np.pi / (points[-1] - points[0] + (points[1]-points[0]))
    dists = fak * cdist(points, points, lambda a, b: a-b)
    np.fill_diagonal(dists, 1e-10)  # to shut up the warning
    if n % 2 == 0:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.tan(0.5 * dists)
    else:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.sin(0.5 * dists)

    np.fill_diagonal(D, 0)
    D *= fak
    phi = points

    def build_residual(rz):
        inshape = rz.shape
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        B = biotsavart.B(compute_derivatives=0)
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        Br = np.cos(phi)*Bx + np.sin(phi)*By
        Bphi = np.cos(phi)*By - np.sin(phi)*Bx
        residual_r = D @ r - r * Br / Bphi
        residual_z = D @ z - r * Bz / Bphi
        return np.vstack((residual_r, residual_z)).reshape(inshape)

    def build_jacobian(rz):
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        B = biotsavart.B(compute_derivatives=1)
        GradB = biotsavart.dB_by_dX()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        dxBx = GradB[:, 0, 0].reshape((n, 1))
        dyBx = GradB[:, 1, 0].reshape((n, 1))
        dzBx = GradB[:, 2, 0].reshape((n, 1))
        dxBy = GradB[:, 0, 1].reshape((n, 1))
        dyBy = GradB[:, 1, 1].reshape((n, 1))
        dzBy = GradB[:, 2, 1].reshape((n, 1))
        dxBz = GradB[:, 0, 2].reshape((n, 1))
        dyBz = GradB[:, 1, 2].reshape((n, 1))
        dzBz = GradB[:, 2, 2].reshape((n, 1))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        Br = cosphi*Bx + sinphi*By
        Bphi = cosphi*By - sinphi*Bx
        drBr = cosphi*cosphi * dxBx + cosphi*sinphi*dyBx + sinphi*cosphi*dxBy + sinphi*sinphi*dyBy
        dzBr = cosphi*dzBx + sinphi*dzBy
        drBphi = cosphi*cosphi*dxBy + cosphi*sinphi*dyBy - sinphi*cosphi*dxBx - sinphi*sinphi*dyBx
        dzBphi = cosphi*dzBy - sinphi*dzBx
        drBz = cosphi * dxBz + sinphi*dyBz
        # residual_r = D @ r - r * Br / Bphi
        # residual_z = D @ z - r * Bz / Bphi
        dr_resr = D + np.diag((-Br/Bphi - r*drBr/Bphi + r*Br*drBphi/Bphi**2).reshape((n,)))
        dz_resr = np.diag((-r*dzBr/Bphi + r*Br*dzBphi/Bphi**2).reshape((n,)))
        dr_resz = np.diag((-Bz/Bphi - r*drBz/Bphi + r*Bz*drBphi/Bphi**2).reshape((n,)))
        dz_resz = D + np.diag((-r*dzBz/Bphi + r*Bz*dzBphi/Bphi**2).reshape((n,)))
        return np.block([[dr_resr, dz_resr], [dr_resz, dz_resz]])
    
    r0 = np.ones_like(phi) * rguess
    z0 = np.zeros_like(phi)
    x0 = np.vstack((r0, z0))
    # h = np.random.rand(x0.size).reshape(x0.shape)
    # eps = 1e-4
    # drdh = build_jacobian(x0)@h
    # drdh_est = (build_residual(x0+eps*h)-build_residual(x0-eps*h))/(2*eps)
    # err = np.linalg.norm(drdh-drdh_est)
    # print(err)
    # print(np.hstack((drdh, drdh_est)))

    # diff = 1e10
    # soln = x0.copy()
    # for i in range(50):
        # r = build_residual(soln)
        # print("r", np.linalg.norm(r))
        # update = np.linalg.solve(build_jacobian(soln), r)
        # soln -= 0.01 * update
        # diff = np.linalg.norm(update)
        # print('dx', diff)
    soln = fsolve(build_residual, x0, fprime=build_jacobian, xtol=1e-13)
    if output == 'cylindrical':
        return np.hstack((soln[:n, None], phi, soln[n:, None]))
    else:
        return np.hstack((soln[:n, None]*np.cos(phi), soln[:n, None]*np.sin(phi), soln[n:, None]))



def full_orbit_rhs_batch(xyzvxyz, m, q, biotsavart, active_idxs):
    bs = xyzvxyz.shape[0]//6
    xyzvxyz = xyzvxyz.reshape((6, bs))
    p = xyzvxyz[np.ix_([0, 1, 2], active_idxs)].T
    v = xyzvxyz[np.ix_([3, 4, 5], active_idxs)]
    biotsavart.set_points(p)
    Bs = biotsavart.B(compute_derivatives=0)
    rhs = np.zeros_like(xyzvxyz)
    rhs[np.ix_([0, 1, 2], active_idxs)] = v
    rhs[np.ix_([3, 4, 5], active_idxs)] = (q/m) * np.cross(v.T, Bs, axis=1).T
    return rhs.flatten()

def guiding_center_rhs_batch(xyzv, vtotal, mus, m, q, biotsavart, active_idxs):
    bs = xyzv.shape[0]//4
    xyzv = xyzv.reshape((4, bs))
    xyzs = xyzv[:3, :].T
    res = np.zeros_like(xyzv)
    biotsavart.set_points(xyzs[active_idxs, :])
    Bs = biotsavart.B(compute_derivatives=1)
    GradBs = biotsavart.dB_by_dX(compute_derivatives=1)
    AbsBs = np.linalg.norm(Bs, axis=1)
    GradAbsBs = (Bs[:, None, 0]*GradBs[:, :, 0] + Bs[:, None, 1]*GradBs[:, :, 1] + Bs[:, None, 2]*GradBs[:, :, 2])/AbsBs[:, None]
    BcrossGradAbsBs = np.asarray([
        (Bs[:, 1] * GradAbsBs[:, 2]) - (Bs[:, 2] * GradAbsBs[:, 1]),
        (Bs[:, 2] * GradAbsBs[:, 0]) - (Bs[:, 0] * GradAbsBs[:, 2]),
        (Bs[:, 0] * GradAbsBs[:, 1]) - (Bs[:, 1] * GradAbsBs[:, 0])]).T

    vtangs = (xyzv[3, :])[active_idxs]
    rhsxyz = (vtangs/AbsBs)[:, None] * Bs
    vperp2s = 2 * mus[active_idxs] * AbsBs
    # vperp2s = vtotal**2 - vtangs**2
    rhsxyz += ((m/(q*AbsBs**3)) * 0.5*vperp2s)[:, None] * BcrossGradAbsBs
    rhsxyz += ((m/(q*AbsBs**3)) * vtangs**2)[:, None] * BcrossGradAbsBs
    rhsv = - mus[active_idxs] * (Bs[:, 0]*GradAbsBs[:, 0] + Bs[:, 1]*GradAbsBs[:, 1] + Bs[:, 2]*GradAbsBs[:, 2])/AbsBs
    res[np.ix_([0, 1, 2], active_idxs)] = rhsxyz.T
    res[np.ix_([3], active_idxs)] = rhsv
    # print('vtotal', np.sqrt(vperp2s + vtangs**2))
    return res.flatten()


def trace_particles_on_axis(axis, biotsavart, nparticles, mode='gyro', tmax=1e-4, seed=1, mass=1.67e-27, charge=1, Ekinev=9000, umin=-1, umax=+1, critical_distance=0.3):
    assert mode in ['gyro', 'orbit']

    e = 1.6e-19
    Ekin = Ekinev * e
    m = mass
    q = charge*e

    vtotal = sqrt(2*Ekin/m) # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    info_all_sync(f"|v| = {vtotal:.2E} m/s")
    max_step = 1.0/vtotal


    np.random.seed(seed)
    xyz_inits = axis[np.random.randint(0, axis.shape[0], size=(nparticles, )), :]

    # us = np.linspace(umin, umax, nparticles, endpoint=True)
    us = np.random.uniform(low=umin, high=umax, size=(nparticles, ))
    # us = np.linspace(0., 1.0, nparticles, endpoint=True)
    # us = +1. * us
    # us = us[6:9]
    # xyz_inits = xyz_inits[6:9,:]
    nparticles = len(us)

    biotsavart.set_points(xyz_inits)
    Bs = biotsavart.B(compute_derivatives=0)
    AbsBs = np.linalg.norm(Bs, axis=1)
    info_all_sync(f"Mean(|B|) = {np.mean(AbsBs):.2E} T")


    # u = 1: all tangential velocity, u = 0: all perpendicular velocity
    info_all_sync("us = %s" % us)
    vtangs = us * vtotal
    vperp2s = vtotal**2 - vtangs**2
    mus = vperp2s/(2*AbsBs)
    from scipy.integrate import solve_ivp, RK45, OdeSolution, DOP853
    tspan = [0, tmax]
    active_idxs = list(range(0, nparticles))
    if mode == 'gyro':
        yinit = np.zeros((4, nparticles))
        yinit[:3, :] = xyz_inits.T
        yinit[3, :] = vtangs
        rhs = lambda t, xyzvs: guiding_center_rhs_batch(xyzvs, vtotal, mus, m, q, biotsavart, active_idxs)
        # solver = RK45(rhs, tspan[0], yinit.flatten(), tspan[-1], max_step=max_step, rtol=1e-6, atol=1e-6)
        solver = DOP853(rhs, tspan[0], yinit.flatten(), tspan[-1], max_step=max_step, rtol=1e-10, atol=1e-10)
    else:
        rhs = lambda t, xyzvyz: full_orbit_rhs_batch(xyzvyz, m, q, biotsavart, active_idxs)
        eB = Bs/AbsBs[:, None]
        ez = np.asarray(nparticles*[[0., 0., -1.]])
        ez -= eB * np.sum(eB*ez, axis=1)[:, None]
        ez *= 1./np.linalg.norm(ez, axis=1)[:, None]
        Bperp = np.cross(eB, ez, axis=1)
        Bperp *= 1./np.linalg.norm(Bperp, axis=1)[:, None]
        yinit = np.zeros((6, nparticles))
        for i in range(nparticles):
            rg = m*sqrt(vperp2s[i])/(abs(q)*AbsBs[i])
            yinit[:3, i] = xyz_inits[i] + rg * ez[i, :]
            yinit[3:, i] = -sqrt(vperp2s[i]) * Bperp[i, :] + vtangs[i] * eB[i, :]
            # solver = RK45(rhs, tspan[0], yinit.flatten(), tspan[-1], max_step=max_step, rtol=1e-6, atol=1e-6)
            solver = DOP853(rhs, tspan[0], yinit.flatten(), tspan[-1], max_step=max_step, rtol=1e-12, atol=1e-12)
    ts = [0]
    denseoutputs = []
    t = tspan[0]
    loss_time = nparticles * [np.inf]
    vsize = 6 if mode == 'orbit' else 4
    while t < tspan[-1]:
        try:
            solver.step()
        except:
            info_all(f'abort (except) at t = {solver.t:.3e}')
            break
        y = solver.y.reshape((vsize, nparticles))
        for i in reversed(range(len(active_idxs))):
            idx = active_idxs[i]
            xyz = y[:3, idx]
            dists = np.linalg.norm(xyz[None, :] - axis, axis=1)
            if min(dists) > critical_distance:
                info_all(f'abort for u={us[idx]:+.3f} (distance) at t={solver.t:.3e}'.replace('+', ' '))
                del active_idxs[i]
                loss_time[idx] = solver.t
        if len(active_idxs) == 0:
            break
        t = solver.t
        ts.append(solver.t)
        denseoutputs.append(solver.dense_output())
    odesol = OdeSolution(ts, denseoutputs)
    N = 10000
    t_eval = np.linspace(ts[0], ts[-1], N)
    res = odesol(t_eval).T.reshape((N, vsize, nparticles))

    res_x = []
    for i in range(nparticles):
        res_x.append(res[:, :3, i])
    return res_x, loss_time, us
