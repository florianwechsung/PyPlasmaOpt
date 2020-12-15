import numpy as np
import cppplasmaopt as cpp
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



def full_orbit_rhs(xyzvxyz, m, q, biotsavart):
    p = xyzvxyz[:3]
    v = xyzvxyz[3:]
    biotsavart.set_points(np.asarray([p]))
    B = biotsavart.B(compute_derivatives=0)[0, :]
    # print('vtang', np.sum(v*B)/np.linalg.norm(B))
    rhs = np.zeros_like(xyzvxyz)
    rhs[:3] = v
    rhs[3:] = (q/m) * np.cross(v, B)
    return rhs

def guiding_center_rhs(xyzv, vtotal, mu, m, q, biotsavart):
    xyz = xyzv[:3]
    vtang = xyzv[3]
    res = np.zeros_like(xyzv)
    biotsavart.set_points(np.asarray([xyz]))
    B = biotsavart.B(compute_derivatives=1)[0, :]
    GradB = biotsavart.dB_by_dX(compute_derivatives=1)[0, :, :]
    AbsB = sqrt(B[0]**2 + B[1]**2 + B[2]**2)
    GradAbsB = (B[0]*GradB[:, 0] + B[1]*GradB[:, 1] + B[2]*GradB[:, 2])/AbsB
    vperp2 = 2 * mu * AbsB
    # vperp2 = vtotal**2 - vtang**2
    term_tang = vtang * B/AbsB
    # print("AbsB", AbsB)
    # print("vtang", vtang)
    # mu = 0.5 * vperp2/AbsB
    # print("mu", mu)
    if vperp2 < 0:
        print("vperp2", vperp2)
        raise RuntimeError
    #BcrossGradAbsB = np.cross(B, GradAbsB) # np.cross is shockingly slow
    BcrossGradAbsB = np.asarray([
        (B[1] * GradAbsB[2]) - (B[2] * GradAbsB[1]),
        (B[2] * GradAbsB[0]) - (B[0] * GradAbsB[2]),
        (B[0] * GradAbsB[1]) - (B[1] * GradAbsB[0])])
    gradB_drift = (m/(q*AbsB**3)) * 0.5*vperp2 * BcrossGradAbsB
    curv_drift = (m/(q*AbsB**3)) * vtang**2 * BcrossGradAbsB
    # curv_drift = (m/(q*AbsB**4)) * vtang**2 * np.cross(B, np.asarray([np.sum(B*GradB[0, ]), np.sum(B*GradB[1, ]), np.sum(B*GradB[2, ])]))
    term_perp = gradB_drift + curv_drift
    res[:3] = term_tang + term_perp
    #res[3] = - mu * np.sum(B * GradAbsB)/AbsB
    res[3] = - mu * (B[0]*GradAbsB[0] + B[1]*GradAbsB[1] + B[2]*GradAbsB[2])/AbsB
    return res

def trace_particles_on_axis(axis, biotsavart, nparticles, mode='gyro', tmax=1e-4):
    assert mode in ['gyro', 'orbit']

    e = 1.6e-19
    # proton
    Ekin = 9 * 1e3 * e
    m = 1.67e-27
    q = e

    vtotal = sqrt(2*Ekin/m) # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    print("|v| = %.2E m/s" % vtotal)
    max_step = 1.0/vtotal


    xyz_init = axis[0, :]
    biotsavart.set_points(np.asarray([xyz_init]))
    B = biotsavart.B(compute_derivatives=0)[0, :]
    AbsB = sqrt(B[0]**2+B[1]**2+B[2]**2)
    print("|B| = %.2E T" % AbsB)

    def solve_for_u(u):
        # u = 1: all tangential velocity, u = 0: all perpendicular velocity
        vtang = u * vtotal
        vperp2 = vtotal**2 - vtang**2
        mu = vperp2/(2*AbsB)

        from scipy.integrate import solve_ivp, RK45, OdeSolution, DOP853
        tspan = [0, tmax]
        if mode == 'gyro':
            yinit = np.asarray([xyz_init[0], xyz_init[1], xyz_init[2], vtang])
            rhs = lambda t, xyz: guiding_center_rhs(xyz, vtotal, mu, m, q, biotsavart)
            # solver = RK45(rhs, tspan[0], yinit, tspan[-1], max_step=max_step, rtol=1e-6, atol=1e-6)
            solver = DOP853(rhs, tspan[0], yinit, tspan[-1], max_step=max_step, rtol=1e-6, atol=1e-6)
        else:
            rhs = lambda t, xyzvyz: full_orbit_rhs(xyzvyz, m, q, biotsavart)
            Bnorm = B/AbsB
            ez = np.asarray([0., 0., 1.])
            ez -= Bnorm * np.sum(Bnorm*ez)
            ez *= 1./np.linalg.norm(ez)
            Bperp = np.cross(Bnorm, ez)
            Bperp *= 1./np.linalg.norm(Bperp)
            rg = m*sqrt(vperp2)/(abs(q)*AbsB)
            yinit = np.zeros((6, ))
            yinit[:3] = xyz_init + rg * ez
            yinit[3:] = -sqrt(vperp2) * Bperp + vtang * Bnorm
            # solver = RK45(rhs, tspan[0], yinit, tspan[-1], max_step=max_step, rtol=1e-6, atol=1e-6)
            solver = DOP853(rhs, tspan[0], yinit, tspan[-1], max_step=max_step, rtol=1e-6, atol=1e-6)
        ts = [0]
        denseoutputs = []
        t = tspan[0]
        while t < tspan[-1]:
            try:
                solver.step()
            except:
                print('abort (except) at t =', solver.t)
                break
            if solver.t < t + 1e-17: # no progress --> abort
                print('abort (timestep) at t =', solver.t)
                break
            xyz = solver.y[:3]
            dists = np.linalg.norm(xyz[None, :] - axis, axis=1)
            if min(dists) > 0.3:
                print('abort (distance) at t =', solver.t)
                break
            t = solver.t
            ts.append(solver.t)
            denseoutputs.append(solver.dense_output())
        odesol = OdeSolution(ts, denseoutputs)
        t_eval = np.linspace(ts[0], ts[-1], 10000)
        res = odesol(t_eval).T
        return res[:, :3], ts[-1]
    res_x = []
    res_t = []
    # for u in [0.10]:
    for u in np.linspace(-1., 1., nparticles, endpoint=True):
        print("u", u)
        tmp = solve_for_u(u)
        res_x.append(tmp[0])
        res_t.append(tmp[1])
    return res_x, res_t
