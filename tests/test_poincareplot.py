import pytest
from pyplasmaopt import BiotSavart, get_24_coil_data, CoilCollection, compute_field_lines

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
            mlab.plot3d(coil.gamma[:, 0], coil.gamma[:, 1], coil.gamma[:, 2])
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
