from pyplasmaopt import *
from math import log10, floor, ceil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)

from example3_get_objective import example3_get_objective
obj, args = example3_get_objective()
outdir = obj.outdir

xiterates = np.loadtxt(outdir + "xiterates.txt")
it = -1
obj.set_dofs(xiterates[it, :])
nperiods = 200
if it == 0:
    magnetic_axis_radius=1.5845
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

