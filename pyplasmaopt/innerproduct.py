import numpy as np
def build_inner_product_matrix_for_curve(curve, derivs):
    x0 = curve.get_dofs()
    n = x0.size
    H = np.zeros((n, n))
    gammas = []
    for i in range(n):
        x = np.zeros_like(x0)
        x[i] = 1.0
        curve.set_dofs(x)
        gammas.append([curve.gamma().copy()])
        if derivs > 0:
            gammas[-1].append(curve.gammadash().copy())
        if derivs > 1:
            gammas[-1].append(curve.gammadashdash().copy())
        if derivs > 2:
            gammas[-1].append(curve.gammadashdashdash().copy())
    curve.set_dofs(x0)

    for i in range(n):
        for j in range(i+1):
            for k in range(derivs+1):
                # H[i, j] += 0.1**k * np.mean(np.sum(gammas[i][k] * gammas[j][k], axis=1))
                H[i, j] += np.mean(np.sum(gammas[i][k] * gammas[j][k], axis=1))
            H[j, i] = H[i, j]
    return H

def build_inner_product_matrices(obj):
    coil_derivs_needed = 1
    if obj.curvature_weight > 0:
        coil_derivs_needed = 2
    if obj.torsion_weight > 0:
        coil_derivs_needed = 3
    coil_derivs_needed = 1
    axis_derivs_needed = 3
    coilmats = [build_inner_product_matrix_for_curve(c, coil_derivs_needed) for c in obj.stellarator._base_coils]
    axismat = build_inner_product_matrix_for_curve(obj.ma, axis_derivs_needed)
    from scipy.linalg import block_diag, sqrtm, eigh


    ms = [np.eye(1), axismat, np.eye(len(coilmats))] + coilmats
    if obj.mode[0:4] == "cvar":
        ms.append(np.eye(1))
    H = block_diag(*ms)
    ms12 = []
    ms12inv = []
    for i in range(len(ms)):
        D, E = eigh(ms[i])
        ms12.append(E @ np.diag(np.abs(D)**0.5) @ E.T)
        ms12inv.append(E @ np.diag(np.abs(D)**-0.5) @ E.T)
    H12 = block_diag(*ms12)
    H12inv = block_diag(*ms12inv)
    return H, H12, H12inv

