from pyplasmaopt import *
from math import log10, floor, ceil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size': 5})
# rc('text', usetex=True)

from example3_get_objective import example3_get_objective
obj, args = example3_get_objective()
outdir = obj.outdir

xiterates = np.loadtxt(outdir + "xiterates.txt")
it = -1
x = xiterates[it, :]
obj.set_dofs(x)

################################################################################
# Hessian ######################################################################
################################################################################

def approx_H(x):
    n = x.size
    H = np.zeros((n, n))
    x0 = x
    eps = 1e-4
    for i in range(n):
        x = x0.copy()
        x[i] += eps
        obj.update(x)
        d1 = obj.dres
        x[i] -= 2*eps
        obj.update(x)
        d0 = obj.dres
        H[i, :] = (d1-d0)/(2*eps)
    H = 0.5 * (H+H.T)
    return H

from scipy.linalg import eigh
H = approx_H(x)
evals = eigh(H, eigvals_only=True) 
sortedabsevals = np.flip(np.sort(np.abs(evals)))
print("Eigenvalues")
print(evals)
np.savetxt(outdir + "evals_it_%i.txt" % it, evals)
np.savetxt(outdir + "sortedabsevals_it_%i.txt" % it, sortedabsevals)
import matplotlib.pyplot as plt
plt.semilogy(sortedabsevals)
plt.xlabel('Index')
plt.ylabel('Magnitude')
plt.title(outdir)
plt.savefig(outdir + 'sortedabsevals_it_%i.png' % it, dpi=600)
plt.close()
evals = eigh(H[obj.coil_dof_idxs[0]:obj.coil_dof_idxs[1], obj.coil_dof_idxs[0]:obj.coil_dof_idxs[1]], eigvals_only=True) 
sortedabsevals = np.flip(np.sort(np.abs(evals)))
print("Eigenvalues (coil only)")
print(evals)
np.savetxt(outdir + "evals_coil_only_it_%i.txt" % it, evals)
np.savetxt(outdir + "sortedabsevals_coil_only_it_%i.txt" % it, sortedabsevals)

################################################################################
# Poincare plot ################################################################
################################################################################

nperiods = 200
if it == 0:
    magnetic_axis_radius=1.5908
else:
    magnetic_axis_radius=obj.ma.gamma[0, 0]

spp = 120
rphiz, xyz, absB, phi_no_mod = compute_field_lines(obj.biotsavart, nperiods=nperiods, batch_size=4, magnetic_axis_radius=magnetic_axis_radius, max_thickness=0.9, delta=0.01, steps_per_period=spp)
nparticles = rphiz.shape[0]

data0 = np.zeros((nperiods, nparticles*2))
data1 = np.zeros((nperiods, nparticles*2))
data2 = np.zeros((nperiods, nparticles*2))
data3 = np.zeros((nperiods, nparticles*2))
for i in range(nparticles):
    data0[:, 2*i+0] = rphiz[i, range(0, nperiods*spp, spp), 0]
    data0[:, 2*i+1] = rphiz[i, range(0, nperiods*spp, spp), 2]
    data1[:, 2*i+0] = rphiz[i, range(1*spp//(obj.ma.nfp*4), nperiods*spp, spp), 0]
    data1[:, 2*i+1] = rphiz[i, range(1*spp//(obj.ma.nfp*4), nperiods*spp, spp), 2]
    data2[:, 2*i+0] = rphiz[i, range(2*spp//(obj.ma.nfp*4), nperiods*spp, spp), 0]
    data2[:, 2*i+1] = rphiz[i, range(2*spp//(obj.ma.nfp*4), nperiods*spp, spp), 2]
    data3[:, 2*i+0] = rphiz[i, range(3*spp//(obj.ma.nfp*4), nperiods*spp, spp), 0]
    data3[:, 2*i+1] = rphiz[i, range(3*spp//(obj.ma.nfp*4), nperiods*spp, spp), 2]

np.savetxt(outdir + 'poincare0_it_%i.txt' % it, data0, comments='', delimiter=',')
np.savetxt(outdir + 'poincare1_it_%i.txt' % it, data1, comments='', delimiter=',')
np.savetxt(outdir + 'poincare2_it_%i.txt' % it, data2, comments='', delimiter=',')
np.savetxt(outdir + 'poincare3_it_%i.txt' % it, data3, comments='', delimiter=',')
modBdata = np.hstack((phi_no_mod[:, None], absB.T))[0:(10*spp)]
np.savetxt(outdir + 'modB_it_%i.txt' % it, modBdata, comments='', delimiter=',')
plt.figure()
for i in range(min(modBdata.shape[1]-1, 10)):
    plt.plot(modBdata[:, 0], modBdata[:, i+1], zorder=100-i)
plt.savefig(outdir + "absB_%i.png" % it, dpi=300)
plt.close()
import mayavi.mlab as mlab
mlab.options.offscreen = True
for coil in obj.stellarator.coils:
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
mlab.savefig(outdir + "poincare-3d_%i.png" % it, magnification=4)
mlab.close()

plt.figure()
for i in range(nparticles):
    plt.scatter(rphiz[i, range(0, nperiods*spp, spp), 0], rphiz[i, range(0, nperiods*spp, spp), 2], s=0.1)
# plt.show()
plt.savefig(outdir + "poincare_%i.png" % it, dpi=300)
plt.close()

################################################################################
# Plot coils ###################################################################
################################################################################

import mayavi.mlab as mlab
mlab.options.offscreen = True
mlab.figure(bgcolor=(1, 1, 1))
colors = [
    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
    (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
    (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
    (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
    (0.8, 0.7254901960784313, 0.4549019607843137),
    (0.39215686274509803, 0.7098039215686275, 0.803921568627451)
]
count = 0
for coil in obj.stellarator.coils:
    coil = coil.gamma
    coil = np.vstack((coil, coil[0,:]))
    mlab.plot3d(coil[:, 0], coil[:, 1], coil[:, 2], tube_radius = 0.015, color = colors[count%len(colors)] )
    count +=1
eo = 3
gamma = obj.ma.gamma
theta = 2*np.pi/obj.ma.nfp
rotmat = np.asarray([
    [cos(theta), -sin(theta), 0],
    [sin(theta), cos(theta), 0],
    [0, 0, 1]]).T
gamma0 = gamma.copy()
for i in range(1, obj.ma.nfp):
    gamma0 = gamma0 @ rotmat
    gamma = np.vstack((gamma, gamma0))
mlab.points3d(gamma[::eo, 0], gamma[::eo, 1], gamma[::eo, 2], mode = 'sphere', scale_factor=0.05, color = (0,0,0))
mlab.gcf().scene.parallel_projection = True
mlab.view(azimuth=0, elevation=40)
mlab.savefig(outdir + "coils-it-%i.png" % it, magnification=4)
mlab.clf()
