import pytest
import numpy as np
from pyplasmaopt import CartesianFourierCurve, BiotSavart, StelleratorSymmetricCylindricalFourierCurve, SquaredMagneticFieldNormOnCurve, RotatedCurve, SquaredMagneticFieldGradientNormOnCurve, CoilCollection
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
    dir_path = os.path.dirname(os.path.realpath(__file__))

    outcoils_matt = np.loadtxt(os.path.join(dir_path, "..", "data", "outcoils_matt.dat"), delimiter=',')
    num_coils = 6
    coils = [CartesianFourierCurve(6, np.linspace(0, 1, 80, endpoint=False)) for i in range(num_coils)]
    Nt = 4
    nfp = 2
    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = outcoils_matt[0, 6*ic + 1]
        coils[ic].coefficients[1][0] = outcoils_matt[0, 6*ic + 3]
        coils[ic].coefficients[2][0] = outcoils_matt[0, 6*ic + 5]
        for io in range(0, Nt-1):
            coils[ic].coefficients[0][2*io+1] = outcoils_matt[io+1, 6*ic + 0]
            coils[ic].coefficients[0][2*io+2] = outcoils_matt[io+1, 6*ic + 1]
            coils[ic].coefficients[1][2*io+1] = outcoils_matt[io+1, 6*ic + 2]
            coils[ic].coefficients[1][2*io+2] = outcoils_matt[io+1, 6*ic + 3]
            coils[ic].coefficients[2][2*io+1] = outcoils_matt[io+1, 6*ic + 4]
            coils[ic].coefficients[2][2*io+2] = outcoils_matt[io+1, 6*ic + 5]

    ma = StelleratorSymmetricCylindricalFourierCurve(4, nfp, np.linspace(0, 1, 80, endpoint=False))
    ma.coefficients[0][0] = 1.
    ma.coefficients[0][1] = 0.076574
    ma.coefficients[0][2] = 0.0032607
    ma.coefficients[0][3] = 2.5405e-05

    ma.coefficients[1][0] = -0.07605
    ma.coefficients[1][1] = -0.0031845
    ma.coefficients[1][2] = -3.1852e-05

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
    print(bs.B(points, use_cpp=False), "\n", matlab_res)
    assert np.allclose(bs.B(points, use_cpp=False), matlab_res)

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

