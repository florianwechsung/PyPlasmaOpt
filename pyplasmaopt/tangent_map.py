import numpy as np
import scipy.integrate

class TangentMap():
    def __init__(self, magnetic_axis, biotsavart):
        self.biotsavart = biotsavart
        self.magnetic_axis = magnetic_axis

    def solve_state(self):
        """
        For biotsavart and magnetic_axis objects, compute rotational transform
        from tangent map by solving initial value probme. 
        """
        t_span = (0,2*np.pi)
        y0 = np.array([1,0,0,1])
        out = scipy.integrate.solve_ivp(self.rhs_fun,t_span,y0,
                                vectorized=False,rtol=1e-6,atol=1e-10)
        if (out.status==0):
            M = out.y[...,-1]
            detM = M[0]*M[3] - M[1]*M[2]
            np.testing.assert_allclose(detM,1,rtol=1e-4)
            trM = M[0] + M[3]
            return np.arccos(trM/2)/(2*np.pi)
        else:
            raise RuntimeError('Error ocurred in integration of tangent map.')
        
    def rhs_fun(self,phi,M):
        """
        Computes the RHS of the tangent map ode, e.g. M'(phi) = rhs
        for given phi and M

        Inputs:
            phi (double): toroidal angle for evaluation
            M (1d array (4)): current value of tangent map
        Outputs:
            rhs (1d array (4)): rhs of ode
        """
        self.magnetic_axis.points = np.array([phi/(2*np.pi)])
        self.magnetic_axis.update()
        self.biotsavart.set_points(self.magnetic_axis.gamma)
        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]
        gradB = self.biotsavart.dB_by_dX
        dBXdX = gradB[...,0,0]
        dBXdY = gradB[...,1,0]
        dBXdZ = gradB[...,2,0]
        dBYdX = gradB[...,0,1]
        dBYdY = gradB[...,1,1]
        dBYdZ = gradB[...,2,1]   
        dBZdX = gradB[...,0,2]
        dBZdY = gradB[...,1,2]
        dBZdZ = gradB[...,2,2]
        X = self.biotsavart.points[:,0]
        Y = self.biotsavart.points[:,1]
        Z = self.biotsavart.points[:,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        dBRdR = (X**2*dBXdX + X*Y * (dBYdX + dBXdY) + Y**2 * dBYdY)/(R**2)
        dBPdR = (X*Y * (dBYdY-dBXdX) + X**2 * dBYdX - Y**2 * dBXdY)/(R**2)
        dBZdR =  dBZdX*X/R + dBZdY*Y/R
        dBRdZ =  dBXdZ*X/R + dBYdZ*Y/R
        dBPdZ = -dBXdZ*Y/R + dBYdZ*X/R 
        m11 = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
        m12 = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
        m21 = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
        m22 = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
        return np.squeeze(np.array([m11*M[0] + m12*M[2], m11*M[1] + m12*M[3], 
                                    m21*M[0] + m22*M[2], m21*M[1] + m22*M[3]]))
