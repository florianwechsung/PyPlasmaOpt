import numpy as np
import cppplasmaopt as cpp

def compute_field_lines(biotsavart, nperiods=200, batch_size=8, magnetic_axis_radius=1, max_thickness=0.5, delta=0.01, steps_per_period=100):

    def cylindrical_to_cartesian(rphiz):
        xyz = np.zeros(rphiz.shape)
        xyz[:, 0] = rphiz[:, 0] * np.cos(rphiz[:, 1])
        xyz[:, 1] = rphiz[:, 0] * np.sin(rphiz[:, 1])
        xyz[:, 2] = rphiz[:, 2]
        return xyz

    gammas                 = [coil.gamma for coil in biotsavart.coils]
    dgamma_by_dphis        = [coil.dgamma_by_dphi[:, 0, :] for coil in biotsavart.coils]
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
