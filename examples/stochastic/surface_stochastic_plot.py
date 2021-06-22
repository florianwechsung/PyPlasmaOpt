import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size': 18})
rc('text', usetex=True)
np.set_printoptions(edgeitems=30, linewidth=200, formatter=dict(float=lambda x: "%+.4e" % x))

# d = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}
import os
os.makedirs('surfaces', exist_ok=True)

sigma = "0p01"
sigma_oos = "0p001"
# sigma = "0p003"
def get_dir(n, ig, ntcoils):
    ppp = 20 if ntcoils < 10 else 15
    if n == 1:
        # return f"output-greene/mc13_config-ncsx_mode-deterministic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-0_noutsamples-1024_seed-0_sigma-{sigma}_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/surfaces_v3_5_5/"
        return f"output-greene/mc13_config-ncsx_mode-deterministic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-0_noutsamples-1024_seed-0_sigma-{sigma}_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/surfaces_sigmaoos_{sigma_oos}_5_5_batch/"
    else:
        # return f"output-greene/mc13_config-ncsx_mode-stochastic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-{n}_noutsamples-1024_seed-0_sigma-{sigma}_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/surfaces_v3_5_5/"
        return f"output-greene/mc13_config-ncsx_mode-stochastic_distribution-gaussian_ppp-{ppp}_Nt_ma-4_Nt_coils-{ntcoils}_ninsamples-{n}_noutsamples-1024_seed-0_sigma-{sigma}_tikhonov-0p0_curvature-1e-06_torsion-0p0_sobolev-0p0_carclen-0p0001_clen-1p0_distw-1000p0_ig-{ig}_ip-l2_optim-scipy/surfaces_sigmaoos_{sigma_oos}_5_5_batch/"




target_areas = np.asarray([0.] + [0.6 + i*1.2 for i in range(8)])
npdata = [target_areas[:9]]
colnames = ['target_areas']
for order in [6]:
    for n in [1, 1024]:
        for ig in [0, 1, 2, 3, 4, 5, 6, 7]:
            outdir = get_dir(n, ig, order)
            try:
                is_non_qs = np.load(outdir + "/is_non_qs_L2.npy")
                is_qs = np.load(outdir + "/is_qs_L2.npy")
                is_iotas = np.load(outdir + "/is_iotas.npy")
                oos_non_qs = np.load(outdir + "/oos_non_qs_L2.npy")
                oos_qs = np.load(outdir + "/oos_qs_L2.npy")
                title = f'n_{n}_order_{order}_IG_{ig}_'
                print((title+"oos_non_qs").ljust(32), np.nanmean(oos_non_qs, axis=1)/np.nanmean(oos_qs, axis=1))
                # print((title+"oos_non_qs_std").ljust(28), np.std(oos_non_qs, axis=1))
                print((title+"is_non_qs").ljust(32), is_non_qs/is_qs)
                # print((title+"is_iota").ljust(32), is_iotas)
                # print(np.sum(np.isnan(oos_non_qs), axis=1)/oos_non_qs.shape[1])
                npdata.append(np.nanmean(oos_non_qs, axis=1)/np.nanmean(oos_qs, axis=1))
                npdata.append(is_non_qs/is_qs)
                # npdata.append(is_iotas)
                colnames.append(title+"oos_non_qs")
                colnames.append(title+"is_non_qs")
                # colnames.append(title+"is_iota")
            except Exception as ex:
                # print(ex)
                pass

np.savetxt(f'surfaces/surface_data_{sigma}_{sigma_oos}.txt', np.asarray(npdata).T, header=" ".join(colnames), comments="")
import sys; sys.exit()

import seaborn as sns
import pandas as pd
plt.figure(figsize=(5,5))
sns.set_style('whitegrid')

labels_density = []
data_density = []
ax = plt.gca()
for order in [6]:
    for n in [1, 1024]:
        color = next(ax._get_lines.prop_cycler)['color']
        for ig in [0, 1, 2, 3, 4, 5, 6, 7]:
            outdir = get_dir(n, ig, order)
            try:
                is_iotas = np.load(outdir + "/is_iotas.npy")
                oos_iotas = np.load(outdir + "/oos_iotas.npy")[0, :]
                oos_iotas = oos_iotas[~np.isnan(oos_iotas)]
                print(n, ig, np.mean(oos_iotas), np.std(oos_iotas), oos_iotas)
                p = sns.kdeplot(oos_iotas, color=color)
                d = p.get_lines()[-1].get_data() 
                data_density += d
                labels_density += [f"x_n_{n}_order_{order}_ig_{ig}", f"y_n_{n}_order_{order}_ig_{ig}"]
            except Exception as ex:
                print(ex)
                pass

plt.show()
# np.savetxt("surfaces/iota_density.txt", np.asarray(data_density).T, delimiter=";", header=";".join(labels_density), comments="")
import sys; sys.exit()


n = 1024
order = 6
ig = np.argmin([np.mean(np.load(get_dir(n, ig, order) + "Jvals_outofsample.npy")) for ig in range(8)])
outdir = get_dir(n, ig, order)

it = -1
print(outdir)

import sys
sys.argv = sys.argv[:1] + [str(s) for s in np.loadtxt(outdir + 'argv.txt', dtype=np.dtype('<U26'))] 
from objective_stochastic import stochastic_get_objective
obj, args = stochastic_get_objective()
obj.noutsamples = 1
x = np.load(outdir + "xiterates.npy")[it, :]
print(f'||dJ||={np.load(outdir + "dJvals.npy")[it, 0]}')
obj.outdir = outdir
obj.update(x)

coils = obj.stellarator.coils
currents = obj.stellarator.currents
ma = obj.ma
iota = obj.qsf.iota

from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
mpol = 5
ntor = 5
nfp = 3
phis_full = np.linspace(0, 1, 5*(2*(nfp*ntor)+2), endpoint=True)
thetas_full = np.linspace(0, 1, 5*(2*(mpol)+2), endpoint=True)
sfull = SurfaceXYZTensorFourier(mpol=mpol, ntor=nfp*ntor, nfp=1, stellsym=False, quadpoints_phi=phis_full, quadpoints_theta=thetas_full)
phis_partial = np.linspace(0, 1/(2*nfp), 5*(ntor+2), endpoint=True)
thetas_partial = np.linspace(0, 1, 5*(2*(mpol)+2), endpoint=True)
spartial = SurfaceXYZTensorFourier(mpol=mpol, ntor=nfp*ntor, nfp=1, stellsym=False, quadpoints_phi=phis_partial, quadpoints_theta=thetas_partial)

sdofs = np.load(outdir + "surfacedofs.npy")

def magnetic_field_on_surface(s):
    bs = obj.biotsavart
    x = s.gamma()
    B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
    mod_B = np.linalg.norm(B, axis=2)
    return mod_B


import mayavi.mlab as mlab
mlab.options.offscreen = True

violet = (87/255, 6/255, 140/255)
copper =  (72.2/100, 45.1/100, 20/100)
for i in [1, 4, 7]:
    # mlab.figure(bgcolor=(1, 1, 1))
    # for c in obj.stellarator._base_coils:
    #     c.plot_mayavi(show=False, color=copper)
    # sfull.set_dofs(sdofs[i, :])
    # sfull.plot(scalars=magnetic_field_on_surface(sfull), wireframe=False)
    # mlab.view(azimuth=45, elevation=45, distance=8)
    # mlab.savefig(f"surfaces/surfaces_mayavi_angled_{i}.png", magnification=4)
    # mlab.close()

    fig = plt.figure()
    spartial.set_dofs(sdofs[i, :])
    im = plt.contourf(2*np.pi*spartial.quadpoints_phi, 2*np.pi*spartial.quadpoints_theta, magnetic_field_on_surface(spartial).T, cmap='viridis')
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta$')
    plt.title('Magnetic field strength on surface')
    plt.colorbar()
    fig.tight_layout()
    plt.savefig(f'surfaces/surfaces_{i}.pdf')
    plt.close()

import IPython; IPython.embed()
import sys; sys.exit()
