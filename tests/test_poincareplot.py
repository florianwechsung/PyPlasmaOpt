import pytest
from pyplasmaopt import get_24_coil_data, CoilCollection, compute_field_lines, find_magnetic_axis, trace_particles_on_axis
from simsgeo import BiotSavart

# Not a great test, essentially just checks that the code runs
def test_poincareplot(nparticles=12, nperiods=20):
    nfp = 2
    coils, currents, ma, eta_bar = get_24_coil_data(nfp=nfp, ppp=20, at_optimum=True)
    currents = [1e5 * x for x in   [-2.271314992875459, -2.223774477156286, -2.091959078815509, -1.917569373937265, -2.115225147955706, -2.025410501731495]]

    # coils, currents, ma, eta_bar = get_24_coil_data(nfp=nfp, ppp=20, at_optimum=False)
    # currents = 6 * [1]


    coil_collection = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(coil_collection.coils, coil_collection.currents)
    rphiz, xyz, _, _ = compute_field_lines(bs, nperiods=nperiods, batch_size=8, magnetic_axis_radius=1.1, max_thickness=0.6, delta=0.02)
    nparticles = rphiz.shape[0]

    try:
        import mayavi.mlab as mlab
        if not __name__  == "__main__":
            mlab.options.offscreen = True
        for coil in coil_collection.coils:
            mlab.plot3d(coil.gamma()[:, 0], coil.gamma()[:, 1], coil.gamma()[:, 2])
        colors = [
            (0.298, 0.447, 0.690), (0.866, 0.517, 0.321),
            (0.333, 0.658, 0.407), (0.768, 0.305, 0.321),
            (0.505, 0.447, 0.701), (0.576, 0.470, 0.376),
            (0.854, 0.545, 0.764), (0.549, 0.549, 0.549),
            (0.800, 0.725, 0.454), (0.392, 0.709, 0.803)]
        counter = 0
        for i in range(0, nparticles, nparticles//5):
            mlab.plot3d(xyz[i, :, 0], xyz[i, :, 1], xyz[i, :, 2], tube_radius=0.005, color=colors[counter%len(colors)])
            counter += 1
        mlab.view(azimuth=0, elevation=0)
        if __name__  == "__main__":
            mlab.show()
        else:
            mlab.savefig("/tmp/poincare-particle.png", magnification=4)
            mlab.close()
    except ModuleNotFoundError:
        pass

    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(nparticles):
        plt.scatter(rphiz[i, range(0, nperiods*100, 100), 0], rphiz[i, range(0, nperiods*100, 100), 2], s=0.1)
    # plt.show()
    plt.savefig("/tmp/poincare.png", dpi=300)
    plt.close()

    assert True # not quite sure how to test this, so we just check that it runs.

# if __name__ == "__main__":
#     test_poincareplot()


import numpy as np
def lossplot(loss_times, tmax, title):
    print('Fraction of escaped particles:', len([t for t in loss_times if t<np.inf])/len(loss_times))
    sorted_t = np.sort(loss_times)
    sorted_t = [t for t in sorted_t if t < np.inf]
    lost_percentage = [len([c for c in sorted_t if c <= t])/len(loss_times) for t in sorted_t]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogx(sorted_t + [tmax], lost_percentage + [lost_percentage[-1]], drawstyle='steps-post')
    plt.xlabel('Time')
    plt.ylabel('Loss percentage')
    plt.title(title)
    plt.xlim((1e-5, tmax))
    plt.savefig(title.replace(' ', '-') + '.png')
    plt.close()
    # plt.show()

if __name__ == "__main__":
    from pyplasmaopt import get_ncsx_data, plot_stellarator
    import time
    nfp = 3
    (coils, ma, currents) = get_ncsx_data(Nt=4, ppp=15, case='orig')
    title = 'NCSX'
    # (coils, ma, currents) = get_ncsx_data(Nt=4, ppp=15, case='optim_no_reg')
    # currents = [1.5 * c for c in currents]
    # title = 'NCSX optimised without regularisation'



    # nfp = 2
    # (coils, currents, ma, eta_bar) = get_24_coil_data(Nt_coils=5, Nt_ma=3, nfp=nfp, ppp=10, at_optimum=True)
    # (coils, _, ma, eta_bar) = get_24_coil_data(Nt_coils=5, Nt_ma=3, nfp=nfp, ppp=10, at_optimum=False)
    

    coil_collection = CoilCollection(coils, currents, nfp, True)
    bs = BiotSavart(coil_collection.coils, coil_collection.currents)
    rguess = 1.5
    axis = find_magnetic_axis(bs, 100, rguess, output='cartesian')
    tic = time.time()
    tmax = 1e-2
    nparticles = 1000
    seed = 1
    res_gyro, res_gyro_t = trace_particles_on_axis(axis, bs, nparticles, mode='gyro', tmax=tmax, seed=seed)
    toc = time.time()
    lossplot(res_gyro_t, tmax, title + f' (Guiding Center, seed={seed}, {nparticles} particles)')
    print(f"Time for Guiding Center tracing {toc-tic}s")
    # print("res_gyro_t", res_gyro_t)
    # plot_stellarator(coil_collection, extra_data=[axis] + res_gyro)
    import sys; sys.exit()

    res_orbit, res_orbit_t = trace_particles_on_axis(axis, bs, nparticles, mode='orbit', tmax=tmax, seed=seed)
    print("res_orbit_t", res_orbit_t)
    lossplot(res_orbit_t, tmax, title + '-orbit')
    plot_stellarator(coil_collection, extra_data=[axis] + res_gyro + res_orbit)
    import IPython; IPython.embed()
