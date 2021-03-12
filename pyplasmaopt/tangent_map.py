import numpy as np
import scipy.integrate
from pyplasmaopt.biotsavart import BiotSavart

class TangentMap():
    def __init__(self, magnetic_axis, stellarator):
        self.stellarator = stellarator
        self.biotsavart = BiotSavart(stellarator.coils, stellarator.currents)
        self.magnetic_axis = magnetic_axis
        
    def compute_iota(self):
        """
        Compute rotational transform from tangent map. 
        
        Outputs:
            iota (double): value of rotational transform.
        """
        phi = np.array([2*np.pi])
        M = self.solve_state(phi)[...,-1]
        detM = M[0]*M[3] - M[1]*M[2]
        np.testing.assert_allclose(detM,1,rtol=1e-4)
        trM = M[0] + M[3]
        return np.arccos(trM/2)/(2*np.pi)

    def solve_state(self,phi,rtol=1e-12,atol=1e-12):
        """
        For biotsavart and magnetic_axis objects, compute rotational transform
            from tangent map by solving initial value problem.
        
        Inputs:
            phi (1d array): 1d array for evaluation of tangent map
            rtol (double): relative tolerance for integration
            atol (double): absolute tolerance for integration
        Outputs:
            y (2d array (4,len(phi))): flattened tangent map on grid of toroidal angle
        """
        t_span = (0,2*np.pi)
        y0 = np.array([1,0,0,1])
        out = scipy.integrate.solve_ivp(self.rhs_fun,t_span,y0,
                                vectorized=False,rtol=rtol,atol=atol,t_eval=phi)
        if (out.status==0):
            return out.y
        else:
            raise RuntimeError('Error ocurred in integration of tangent map.')
            
    def compute_m(self,phi):
        """
        Computes the matrix that appears on the rhs of the tangent map ODE, 
            e.g. M'(phi) = m(phi) rhs, for given phi. 

        Inputs:
            phi (double): toroidal angle for evaluation
        Outputs:
            m (1d array (4)): matrix appearing on rhs of tangent map ODE
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
        m = np.zeros((4))
        m[0] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
        m[1] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
        m[2] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
        m[3] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
        return m
        
    def rhs_fun(self,phi,M):
        """
        Computes the RHS of the tangent map ode, e.g. M'(phi) = rhs
            for given phi and M

        Inputs:
            phi (double): toroidal angle for evaluation
            M (1d array (4)): current value of tangent map
        Outputs:
            rhs (1d array (4)): rhs of ODE
        """
        m = self.compute_m(phi)
        return np.squeeze(np.array([m[0]*M[0] + m[1]*M[2], m[0]*M[1] + m[1]*M[3], 
                                    m[2]*M[0] + m[3]*M[2], m[2]*M[1] + m[3]*M[3]]))
    
    def d_iota_dmagneticaxiscoefficients(self,nphi=500):
        """
        Compute derivative of iota wrt axis coefficients
        
        Inputs: 
            nphi (int): number of toroidal grid points for evaluation of integral
                with uniform grid
            
        Outputs:
            d_iota (1d array (ncoeffs)): derivative of iota wrt axis coefficients
        """    
        phi,dphi = np.linspace(2*np.pi,0,nphi,endpoint=False,retstep=True)
        d_m = self.compute_d_m_dmagneticaxiscoefficients(phi)
        lam = self.solve_adjoint_state(phi)
        M = np.flip(self.solve_state(np.flip(phi)),axis=1)
        iota = self.compute_iota()
        fac = -1/(4*np.pi*np.sin(2*np.pi*iota))
        lambda_dot_d_m_times_M = \
              lam[0,:,None]*(d_m[0,...]*M[0,...,None] + d_m[1,...]*M[2,...,None]) \
            + lam[1,:,None]*(d_m[0,...]*M[1,...,None] + d_m[1,...]*M[3,...,None]) \
            + lam[2,:,None]*(d_m[2,...]*M[0,...,None] + d_m[3,...]*M[2,...,None]) \
            + lam[3,:,None]*(d_m[2,...]*M[1,...,None] + d_m[3,...]*M[3,...,None])
        d_iota = -fac*np.sum(lambda_dot_d_m_times_M,axis=(0))*dphi
    
        return d_iota
    
    def d_iota_dcoilcurrents(self,nphi=500):
        """
        Compute derivative of iota wrt coil currents.
        
        Inputs: 
            nphi (int): number of toroidal grid points for evaluation of integral
                with uniform grid.
            
        Outputs:
            d_iota (list of 1d arrays (ncurrents)): derivative of iota wrt 
                coil currents 
        """    
        phi,dphi = np.linspace(2*np.pi,0,nphi,endpoint=False,retstep=True)
        d_m_by_dcoilcurrents = self.compute_d_m_dcoilcurrents(phi)
        lam = self.solve_adjoint_state(phi)
        M = np.flip(self.solve_state(np.flip(phi)),axis=1)
        iota = self.compute_iota()
        fac = -1/(4*np.pi*np.sin(2*np.pi*iota))
        d_iota_dcoilcurrents = []
        for i in range(len(d_m_by_dcoilcurrents)):
            d_m = d_m_by_dcoilcurrents[i]
            lambda_dot_d_m_times_M = \
                  lam[0,:]*(d_m[0,...]*M[0,...] + d_m[1,...]*M[2,...]) \
                + lam[1,:]*(d_m[0,...]*M[1,...] + d_m[1,...]*M[3,...]) \
                + lam[2,:]*(d_m[2,...]*M[0,...] + d_m[3,...]*M[2,...]) \
                + lam[3,:]*(d_m[2,...]*M[1,...] + d_m[3,...]*M[3,...])
            d_iota = -fac*np.sum(lambda_dot_d_m_times_M)*dphi
            d_iota_dcoilcurrents.append(d_iota)
        d_iota_dcoilcurrents = \
            self.stellarator.reduce_current_derivatives([ires for ires in d_iota_dcoilcurrents])
    
        return d_iota_dcoilcurrents

    def d_iota_dcoilcoeffs(self,nphi=500):
        """
        Compute derivative of iota wrt coil coeffs.
        
        Inputs: 
            nphi (int): number of toroidal grid points for evaluation of integral
                on uniform grid 
            
        Outputs:
            d_iota (list of 1d arrays (ncoeffs)): derivatives of iota wrt
                coil coefficients
        """
        phi,dphi = np.linspace(2*np.pi,0,nphi,endpoint=False,retstep=True)
        d_m_by_dcoilcoeffs = self.compute_d_m_dcoilcoeffs(phi)
        lam = self.solve_adjoint_state(phi)
        M = np.flip(self.solve_state(np.flip(phi)),axis=1)
        iota = self.compute_iota()
        fac = -1/(4*np.pi*np.sin(2*np.pi*iota))
        d_iota_dcoilcoeffs = []
        for i in range(len(d_m_by_dcoilcoeffs)):
            d_m = d_m_by_dcoilcoeffs[i]
            lambda_dot_d_m_times_M = \
                  lam[0,:,None]*(d_m[0,...]*M[0,...,None] + d_m[1,...]*M[2,...,None]) \
                + lam[1,:,None]*(d_m[0,...]*M[1,...,None] + d_m[1,...]*M[3,...,None]) \
                + lam[2,:,None]*(d_m[2,...]*M[0,...,None] + d_m[3,...]*M[2,...,None]) \
                + lam[3,:,None]*(d_m[2,...]*M[1,...,None] + d_m[3,...]*M[3,...,None])
            d_iota = -fac*np.sum(lambda_dot_d_m_times_M,axis=(0))*dphi
            d_iota_dcoilcoeffs.append(d_iota)
        d_iota_dcoilcoeffs = self.stellarator.reduce_coefficient_derivatives([ires for ires in d_iota_dcoilcoeffs])
        return d_iota_dcoilcoeffs
    
    def solve_adjoint_state(self,phi,rtol=1e-12,atol=1e-12):
        """
        For biotsavart and magnetic_axis objects, compute adjoint variable 
            by solving initial value probme. 
                
        Inputs:
            phi (1d array): toroidal angle for evaluation of adjoint variable
        """
        t_span = (2*np.pi,0)
        y0 = np.array([1,0,0,1])
        out = scipy.integrate.solve_ivp(self.adjoint_rhs_fun,t_span,y0,
                                vectorized=False,rtol=rtol,atol=atol,
                                       t_eval=phi)
        if (out.status==0):
            return out.y
        else:
            raise RuntimeError('Error ocurred in integration of adjoint tangent map.')
            
    def adjoint_rhs_fun(self,phi,M):
        """
        Computes the RHS of the adjoint tangent map ODE, e.g. lambda'(phi) = rhs
            for given phi and lambda

        Inputs:
            phi (double): toroidal angle for evaluation
            lambda (1d array (4)): current value of adjoint tangent map
        Outputs:
            rhs (1d array (4)): rhs of ode
        """
        m = self.compute_m(phi)
        
        return -np.squeeze(np.array([m[0]*M[0] + m[2]*M[2], m[0]*M[1] + m[2]*M[3], 
                                     m[1]*M[0] + m[3]*M[2], m[1]*M[1] + m[3]*M[3]]))

    def compute_d_m_dcoilcoeffs(self,phi):
        """
        Computes the derivative of  matrix that appears on the rhs of the tangent map ode, 
        e.g. M'(phi) = m(phi) rhs, with respect to coil coeffs for given phi. 

        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_m_dcoilcoeffs (list (ncoils) of 3d array (npoints,4,ncoeffs)): derivative 
                of matrix appearing on rhs on ode wrt coil coeffs
        """
        points = phi/(2*np.pi)
        self.magnetic_axis.points = points
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
        
        R = R[:,None]
        BR = BR[:,None]
        BZ = BZ[:,None]
        BP = BP[:,None]
        X = X[:,None]
        Y = Y[:,None]
        dBRdR = dBRdR[:,None]
        dBPdR = dBPdR[:,None]
        dBZdR = dBZdR[:,None]
        dBRdZ = dBRdZ[:,None]
        dBPdZ = dBPdZ[:,None]
        dBZdZ = dBZdZ[:,None]
       
        # Shape: (ncoils,npoints,nparams,3)
        dB_by_dcoilcoeffs = self.biotsavart.dB_by_dcoilcoeffs
        dgradB_by_dcoilcoeffs = self.biotsavart.d2B_by_dXdcoilcoeffs
        
        d_m_by_dcoilcoeffs = []
        for i in range(len(dB_by_dcoilcoeffs)):
            d_B = dB_by_dcoilcoeffs[i]
            d_BX = d_B[...,0]
            d_BY = d_B[...,1]
            d_BZ = d_B[...,2]
        
            d_gradB = dgradB_by_dcoilcoeffs[i]
            d_dBXdX = d_gradB[...,0,0]
            d_dBXdY = d_gradB[...,1,0]
            d_dBXdZ = d_gradB[...,2,0]
            d_dBYdX = d_gradB[...,0,1]
            d_dBYdY = d_gradB[...,1,1]
            d_dBYdZ = d_gradB[...,2,1]   
            d_dBZdX = d_gradB[...,0,2]
            d_dBZdY = d_gradB[...,1,2]
            d_dBZdZ = d_gradB[...,2,2]        
        
            d_BR =  X * d_BX/R + Y * d_BY/R
            d_dBRdR = (X**2*d_dBXdX 
                    + X*Y * (d_dBYdX + d_dBXdY) 
                    + Y**2 * d_dBYdY)/(R**2)
            d_BP = -Y * d_BX/R + X * d_BY/R
            d_dBPdR = (X*Y * (d_dBYdY-d_dBXdX) 
                    + X**2 * d_dBYdX 
                    - Y**2 * d_dBXdY)/(R**2)
            d_dBZdR =  d_dBZdX*X/R + d_dBZdY*Y/R
            d_dBRdZ =  d_dBXdZ*X/R + d_dBYdZ*Y/R
            d_dBPdZ = -d_dBXdZ*Y/R + d_dBYdZ*X/R
        
            d_m = np.zeros((4,np.shape(d_BR)[0],np.shape(d_BR)[1]))
#             d_m[...,0] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
            d_m[0,...] = d_BR/BP - BR*d_BP/(BP*BP) \
                + R*(d_dBRdR/BP - dBRdR*d_BP/BP**2 - d_BR*dBPdR/BP**2
                    - BR*d_dBPdR/BP**2 + 2*BR*dBPdR*d_BP/(BP**3))
#             d_m[...,1] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
            d_m[1,...] = R*(d_dBRdZ/BP - dBRdZ*d_BP/BP**2 - d_BR*dBPdZ/BP**2
                    - BR*d_dBPdZ/BP**2 + 2*BR*dBPdZ*d_BP/BP**3)
#             d_m[...,2] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
            d_m[2,...] = d_BZ/BP - BZ*d_BP/BP**2 \
                + R*(d_dBZdR/BP - dBZdR*d_BP/BP**2 - d_BZ*dBPdR/BP**2
                    - BZ*d_dBPdR/BP**2 + 2*BZ*dBPdR*d_BP/(BP**3))
#             d_m[...,3] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
            d_m[3,...] = R*(d_dBZdZ/BP - dBZdZ*d_BP/BP**2 
                    - d_BZ*dBPdZ/BP**2 - BZ*d_dBPdZ/BP**2 + 2*BZ*dBPdZ*d_BP/BP**3)
            d_m_by_dcoilcoeffs.append(d_m)
            
        return d_m_by_dcoilcoeffs
    
    def compute_d_m_dcoilcurrents(self,phi):
        """
        Computes the derivative of  matrix that appears on the rhs of the tangent map ode, 
            e.g. M'(phi) = m(phi) rhs, with respect to coil coeffs for given phi. 

        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_m_dcoilcurrents (list (ncoils) of 2d array (4,npoints)): derivative 
                of matrix appearing on rhs on ode wrt coil currents
        """
        points = phi/(2*np.pi)
        self.magnetic_axis.points = points
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
       
        # Shape: (ncoils,npoints,3)
        dB_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents
        dgradB_by_dcoilcurrents = self.biotsavart.d2B_by_dXdcoilcurrents
        
        d_m_by_dcoilcurrents = []
        for i in range(len(dB_by_dcoilcurrents)):
            d_B = dB_by_dcoilcurrents[i]
            d_BX = d_B[...,0]
            d_BY = d_B[...,1]
            d_BZ = d_B[...,2]
        
            d_gradB = dgradB_by_dcoilcurrents[i]
            d_dBXdX = d_gradB[...,0,0]
            d_dBXdY = d_gradB[...,1,0]
            d_dBXdZ = d_gradB[...,2,0]
            d_dBYdX = d_gradB[...,0,1]
            d_dBYdY = d_gradB[...,1,1]
            d_dBYdZ = d_gradB[...,2,1]   
            d_dBZdX = d_gradB[...,0,2]
            d_dBZdY = d_gradB[...,1,2]
            d_dBZdZ = d_gradB[...,2,2]        
        
            d_BR =  (X * d_BX + Y * d_BY)/R
            d_dBRdR = (X**2*d_dBXdX 
                    + X*Y * (d_dBYdX + d_dBXdY) 
                    + Y**2 * d_dBYdY)/(R**2)
            d_BP = (-Y * d_BX + X * d_BY)/R
            d_dBPdR = (X*Y * (d_dBYdY-d_dBXdX) 
                    + X**2 * d_dBYdX 
                    - Y**2 * d_dBXdY)/(R**2)
            d_dBZdR =  d_dBZdX*X/R + d_dBZdY*Y/R
            d_dBRdZ =  d_dBXdZ*X/R + d_dBYdZ*Y/R
            d_dBPdZ = -d_dBXdZ*Y/R + d_dBYdZ*X/R
        
            d_m = np.zeros((4,np.shape(d_BR)[0]))
#             d_m[...,0] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
            d_m[0,...] = d_BR/BP - BR*d_BP/BP**2 \
                + R*(d_dBRdR/BP - dBRdR*d_BP/BP**2 - d_BR*dBPdR/BP**2
                    - BR*d_dBPdR/BP**2 + 2*BR*dBPdR*d_BP/BP**3)
#             d_m[...,1] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
            d_m[1,...] = R*(d_dBRdZ/BP - dBRdZ*d_BP/BP**2 - d_BR*dBPdZ/BP**2
                    - BR*d_dBPdZ/BP**2 + 2*BR*dBPdZ*d_BP/BP**3)
#             d_m[...,2] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
            d_m[2,...] = d_BZ/BP - BZ*d_BP/BP**2 \
                + R*(d_dBZdR/BP - dBZdR*d_BP/BP**2 - d_BZ*dBPdR/BP**2
                    - BZ*d_dBPdR/BP**2 + 2*BZ*dBPdR*d_BP/(BP**3))
#             d_m[...,3] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
            d_m[3,...] = R*(d_dBZdZ/BP - dBZdZ*d_BP/BP**2 
                    - d_BZ*dBPdZ/BP**2 - BZ*d_dBPdZ/BP**2 + 2*BZ*dBPdZ*d_BP/BP**3)
            d_m_by_dcoilcurrents.append(d_m)
            
        return d_m_by_dcoilcurrents
    
    def compute_d_m_dmagneticaxiscoefficients(self,phi):
        """
        Computes the derivative of  matrix that appears on the rhs of the tangent map ode, 
            e.g. M'(phi) = m(phi) rhs, with respect to axis coefficients for given phi. 

        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_m_dcoilcoeffs (3d array (npoints,ncoeffs,4)): derivative 
                of matrix appearing on rhs on ode wrt axis coeffs
        """
        points = phi/(2*np.pi)
        self.magnetic_axis.points = points
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
        
        X = X[:,None]
        Y = Y[:,None]
        R = R[:,None]
        BP = BP[:,None]
        BR = BR[:,None]
        BZ = BZ[:,None]
        BX = BX[:,None]
        BY = BY[:,None]
        dBRdR = dBRdR[:,None]
        dBPdR = dBPdR[:,None]
        dBZdR = dBZdR[:,None]
        dBRdZ = dBRdZ[:,None]
        dBPdZ = dBPdZ[:,None]
        dBZdZ = dBZdZ[:,None] 
        dBXdX = dBXdX[:,None]
        dBXdY = dBXdY[:,None]
        dBXdZ = dBXdZ[:,None]
        dBYdX = dBYdX[:,None]
        dBYdY = dBYdY[:,None]
        dBYdZ = dBYdZ[:,None]
        dBZdX = dBZdX[:,None]
        dBZdY = dBZdY[:,None]
       
        # Shape: (len(self.points), self.num_coeff(), 3)
        dgamma_by_dcoeff  = self.magnetic_axis.dgamma_by_dcoeff
        
        d_X = dgamma_by_dcoeff[...,0]
        d_Y = dgamma_by_dcoeff[...,1]
        d_R = dgamma_by_dcoeff[...,0]*X/R + dgamma_by_dcoeff[...,1]*Y/R
        d_BX = dgamma_by_dcoeff[...,0]*gradB[...,None,0,0] \
             + dgamma_by_dcoeff[...,1]*gradB[...,None,1,0] \
             + dgamma_by_dcoeff[...,2]*gradB[...,None,2,0] 
        d_BY = dgamma_by_dcoeff[...,0]*gradB[...,None,0,1] \
             + dgamma_by_dcoeff[...,1]*gradB[...,None,1,1] \
             + dgamma_by_dcoeff[...,2]*gradB[...,None,2,1] 
        d_BZ = dgamma_by_dcoeff[...,0]*gradB[...,None,0,2] \
             + dgamma_by_dcoeff[...,1]*gradB[...,None,1,2] \
             + dgamma_by_dcoeff[...,2]*gradB[...,None,2,2] 
        
        # Shape: ((len(points), 3, 3, 3))
        d2Bbs_by_dXdX = self.biotsavart.d2B_by_dXdX
        
        d_dBXdX = dgamma_by_dcoeff[...,0]*d2Bbs_by_dXdX[...,None,0,0,0] \
            +     dgamma_by_dcoeff[...,1]*d2Bbs_by_dXdX[...,None,1,0,0] \
            +     dgamma_by_dcoeff[...,2]*d2Bbs_by_dXdX[...,None,2,0,0] 
        d_dBXdY = dgamma_by_dcoeff[...,0]*d2Bbs_by_dXdX[...,None,0,1,0] \
            +     dgamma_by_dcoeff[...,1]*d2Bbs_by_dXdX[...,None,1,1,0] \
            +     dgamma_by_dcoeff[...,2]*d2Bbs_by_dXdX[...,None,2,1,0] 
        d_dBXdZ = dgamma_by_dcoeff[...,0]*d2Bbs_by_dXdX[...,None,0,2,0] \
            +     dgamma_by_dcoeff[...,1]*d2Bbs_by_dXdX[...,None,1,2,0] \
            +     dgamma_by_dcoeff[...,2]*d2Bbs_by_dXdX[...,None,2,2,0] 
        d_dBYdX = dgamma_by_dcoeff[...,0]*d2Bbs_by_dXdX[...,None,0,0,1] \
            +     dgamma_by_dcoeff[...,1]*d2Bbs_by_dXdX[...,None,1,0,1] \
            +     dgamma_by_dcoeff[...,2]*d2Bbs_by_dXdX[...,None,2,0,1] 
        d_dBYdY = dgamma_by_dcoeff[...,0]*d2Bbs_by_dXdX[...,None,0,1,1] \
            +     dgamma_by_dcoeff[...,1]*d2Bbs_by_dXdX[...,None,1,1,1] \
            +     dgamma_by_dcoeff[...,2]*d2Bbs_by_dXdX[...,None,2,1,1] 
        d_dBYdZ = dgamma_by_dcoeff[...,0]*d2Bbs_by_dXdX[...,None,0,2,1] \
            +     dgamma_by_dcoeff[...,1]*d2Bbs_by_dXdX[...,None,1,2,1] \
            +     dgamma_by_dcoeff[...,2]*d2Bbs_by_dXdX[...,None,2,2,1]  
        d_dBZdX = dgamma_by_dcoeff[...,0]*d2Bbs_by_dXdX[...,None,0,0,2] \
            +     dgamma_by_dcoeff[...,1]*d2Bbs_by_dXdX[...,None,1,0,2] \
            +     dgamma_by_dcoeff[...,2]*d2Bbs_by_dXdX[...,None,2,0,2] 
        d_dBZdY = dgamma_by_dcoeff[...,0]*d2Bbs_by_dXdX[...,None,0,1,2] \
            +     dgamma_by_dcoeff[...,1]*d2Bbs_by_dXdX[...,None,1,1,2] \
            +     dgamma_by_dcoeff[...,2]*d2Bbs_by_dXdX[...,None,2,1,2] 
        d_dBZdZ = dgamma_by_dcoeff[...,0]*d2Bbs_by_dXdX[...,None,0,2,2] \
            +     dgamma_by_dcoeff[...,1]*d2Bbs_by_dXdX[...,None,1,2,2] \
            +     dgamma_by_dcoeff[...,2]*d2Bbs_by_dXdX[...,None,2,2,2]       

        d_BR =  (X * d_BX + d_X * BX + Y * d_BY + d_Y * BY)/R \
            - BR*d_R/R
        d_dBRdR = (X**2*d_dBXdX + 2* X * d_X * dBXdX
                + X*Y * (d_dBYdX + d_dBXdY) + (d_X * Y + d_Y * X)*(dBYdX + dBXdY)
                + Y**2 * d_dBYdY + 2*Y*d_Y * dBYdY)/(R**2) \
            - 2*dBRdR*d_R/R
        d_BP = (-Y * d_BX - d_Y * BX + X * d_BY + d_X * BY)/R \
            - BP*d_R/R
        d_dBPdR = (X*Y * (d_dBYdY-d_dBXdX) + (d_X*Y + X*d_Y)*(dBYdY-dBXdX)
                + X**2 * d_dBYdX + 2 * X * d_X * dBYdX
                - Y**2 * d_dBXdY - 2 * Y * d_Y * dBXdY)/(R**2) \
            - 2*dBPdR * d_R/R
        d_dBZdR =  (d_dBZdX*X + dBZdX*d_X + d_dBZdY*Y + dBZdY*d_Y)/R - dBZdR*d_R/R
        d_dBRdZ =  (d_dBXdZ*X + dBXdZ*d_X + d_dBYdZ*Y + dBYdZ*d_Y)/R - dBRdZ*d_R/R
        d_dBPdZ =  (-d_dBXdZ*Y - dBXdZ*d_Y + d_dBYdZ*X + dBYdZ*d_X)/R - dBPdZ*d_R/R

        d_m = np.zeros((4,np.shape(d_BR)[0],np.shape(d_BR)[1]))
#             d_m[...,0] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
        d_m[0,...] = d_BR/BP - BR*d_BP/BP**2 \
            + R*(d_dBRdR/BP - dBRdR*d_BP/BP**2 - d_BR*dBPdR/BP**2
                - BR*d_dBPdR/BP**2 + 2*BR*dBPdR*d_BP/BP**3) \
            + d_R * (dBRdR/BP - BR*dBPdR/BP**2)
#             d_m[...,1] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
        d_m[1,...] = R*(d_dBRdZ/BP - dBRdZ*d_BP/BP**2 - d_BR*dBPdZ/BP**2
                - BR*d_dBPdZ/BP**2 + 2*BR*dBPdZ*d_BP/BP**3) \
            + d_R * (dBRdZ/BP - BR*dBPdZ/BP**2)
#             d_m[...,2] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
        d_m[2,...] = d_BZ/BP - BZ*d_BP/BP**2 \
            + R*(d_dBZdR/BP - dBZdR*d_BP/BP**2 - d_BZ*dBPdR/BP**2
                - BZ*d_dBPdR/BP**2 + 2*BZ*dBPdR*d_BP/(BP**3)) \
            + d_R * (dBZdR/BP - BZ*dBPdR/BP**2)
#             d_m[...,3] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
        d_m[3,...] = R*(d_dBZdZ/BP - dBZdZ*d_BP/BP**2 
                - d_BZ*dBPdZ/BP**2 - BZ*d_dBPdZ/BP**2 + 2*BZ*dBPdZ*d_BP/BP**3) \
                + d_R * (dBZdZ/BP - BZ*dBPdZ/BP**2)
            
        return d_m
