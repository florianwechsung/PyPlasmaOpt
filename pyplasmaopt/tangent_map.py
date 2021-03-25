import numpy as np
import scipy.integrate
from pyplasmaopt.biotsavart import BiotSavart

class TangentMap():
    def __init__(self, stellarator, magnetic_axis=None, rtol=1e-12, atol=1e-12,
                constrained=True,tol=1e-5,max_nodes=100000,verbose=0):
        """
        rtol (double): relative tolerance for IVP
        atol (double): absolute tolerance for IVP
        tol (double): tolerance for BVP
        maxnodes (int): maximum nodes for BVP
        constrained (bool): if true, "true" magnetic axis is computed
        verbose (int): verbosity for BVP
        """
        self.stellarator = stellarator
        self.biotsavart = BiotSavart(stellarator.coils, stellarator.currents)
        self.magnetic_axis = magnetic_axis
        self.rtol = rtol
        self.atol = atol
        self.tol = tol
        self.max_nodes = max_nodes
        self.constrained = constrained
        self.verbose = verbose
        self.nphi = 100
        # Polynomial solutions for current solutions
        self.axis_poly = None
        self.tangent_poly = None
        self.adjoint_axis_poly = None
        self.adjoint_tangent_poly = None

    def update_solutions(self):
        phi = np.linspace(0,2*np.pi,self.nphi,endpoint=False)
        phi_reverse = np.linspace(2*np.pi,0,self.nphi,endpoint=False)
        if (self.constrained):
            sol, self.axis_poly = self.compute_axis(phi)  
            sol, self.tangent_poly = self.compute_tangent(phi,self.axis_poly)
            sol, self.adjoint_tangent_poly = self.compute_adjoint_tangent(phi_reverse,
                                                                 self.axis_poly)
            sol, self.adjoint_axis_poly = self.compute_adjoint_axis(phi,
                     self.axis_poly,self.tangent_poly,self.adjoint_tangent_poly)
        else:
            sol, self.axis_poly = self.compute_axis(phi)  
            sol, self.tangent_poly = self.compute_tangent(phi)
            sol, self.adjoint_tangent_poly = self.compute_adjoint_tangent(phi_reverse)
            sol, self.adjoint_axis_poly = self.compute_adjoint_axis(phi,
                                                                 self.axis_poly)

    def reset_solutions(self):
        self.axis_poly = None
        self.tangent_poly = None
        self.adjoint_axis_poly = None
        self.adjoint_tangent_poly = None
        
    def compute_iota(self):
        """
        Compute rotational transform from tangent map. 
        
        Outputs:
            iota (double): value of rotational transform.
        """
        phi = np.array([2*np.pi])
        if self.tangent_poly is None:
            self.update_solutions()
        M = self.tangent_poly(phi)
        detM = M[0]*M[3] - M[1]*M[2]
        np.testing.assert_allclose(detM,1,rtol=1e-2)
        trM = M[0] + M[3]
        return np.arccos(trM/2)/(2*np.pi)

    def compute_tangent(self,phi,axis_poly=None):
        """
        For biotsavart and magnetic_axis objects, compute rotational transform
            from tangent map by solving initial value problem.
        
        Inputs:
            phi (1d array): 1d array for evaluation of tangent map
            axis_poly: polynomial solution for axis
        Outputs:
            y (2d array (4,len(phi))): flattened tangent map on grid of toroidal angle
        """
        if self.constrained:
            if axis_poly is None:
                axis, axis_poly = self.compute_axis(2*np.pi*self.magnetic_axis.nfp*self.magnetic_axis.points)
            args = (axis_poly,)
        else:
            args = None
            
        y0 = np.array([1,0,0,1])
        t_span = (0,2*np.pi)
        out = scipy.integrate.solve_ivp(self.rhs_fun,t_span,y0,
                            vectorized=False,rtol=self.rtol,atol=self.atol,
                                        t_eval=phi,args=args,dense_output=True)
        if (out.status==0):
            return out.y, out.sol
        else:
            raise RuntimeError('Error ocurred in integration of tangent map.')
            
    def compute_m(self,phi,axis=None):
        """
        Computes the matrix that appears on the rhs of the tangent map ODE, 
            e.g. M'(phi) = m(phi) rhs, for given phi. 

        Inputs:
            phi (double): toroidal angle for evaluation
            axis (2d array (2,npoints)): R and Z for current axis state
        Outputs:
            m (1d array (4)): matrix appearing on rhs of tangent map ODE
        """
        if (axis is not None):
            if (np.ndim(axis)>1):
                gamma = np.zeros((len(axis[0,:]),3))
            else:
                gamma = np.zeros((1,3))
            gamma[...,0] = axis[0,...]*np.cos(phi)
            gamma[...,1] = axis[0,...]*np.sin(phi)
            gamma[...,2] = axis[1,...]
            self.biotsavart.set_points(gamma)
        else:
            points = phi/(2*np.pi)
            if (np.ndim(points)==0):
                points = np.array([points])
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
        X = self.biotsavart.points[...,0]
        Y = self.biotsavart.points[...,1]
        Z = self.biotsavart.points[...,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        dBRdR = (X**2*dBXdX + X*Y * (dBYdX + dBXdY) + Y**2 * dBYdY)/(R**2)
        dBPdR = (X*Y * (dBYdY-dBXdX) + X**2 * dBYdX - Y**2 * dBXdY)/(R**2)
        dBZdR =  dBZdX*X/R + dBZdY*Y/R
        dBRdZ =  dBXdZ*X/R + dBYdZ*Y/R
        dBPdZ = -dBXdZ*Y/R + dBYdZ*X/R 
        if (np.ndim(phi)==0):
            m = np.zeros((4,1))
        else:
            m = np.zeros((4,len(phi)))
        m[0,...] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
        m[1,...] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
        m[2,...] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
        m[3,...] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
        return np.squeeze(m)
        
    def rhs_fun(self,phi,M,axis_poly=None):
        """
        Computes the RHS of the tangent map ode, e.g. M'(phi) = rhs
            for given phi and M

        Inputs:
            phi (double): toroidal angle for evaluation
            M (1d array (4)): current value of tangent map
            axis_poly (instance of scipy.interpolate.PPoly cubic spline): polynomial
                representing magnetic axis solution
        Outputs:
            rhs (1d array (4)): rhs of ODE
        """
        if (axis_poly is not None):
            m = self.compute_m(phi,axis_poly(phi))
        else:
            m = self.compute_m(phi)
        out = np.squeeze(np.array([m[0]*M[0] + m[1]*M[2], m[0]*M[1] + m[1]*M[3], 
                                   m[2]*M[0] + m[3]*M[2], m[2]*M[1] + m[3]*M[3]]))
        return out
    
    def d_iota_d_magneticaxiscoeffs(self,nphi=500):
        """
        Compute derivative of iota wrt axis coefficients
        
        Inputs: 
            nphi (int): number of toroidal grid points for evaluation of integral
                with uniform grid
            
        Outputs:
            d_iota (1d array (ncoeffs)): derivative of iota wrt axis coefficients
        """    
        phi, dphi = np.linspace(2*np.pi,0,nphi,endpoint=False,retstep=True)
        # Update solutions if necessary
        if (self.tangent_poly is None or self.adjoint_tangent_poly is None):
            self.update_solutions()
            
        d_m = self.compute_d_m_d_magneticaxiscoeffs(phi)
        lam = self.adjoint_tangent_poly(phi)
        M = self.tangent_poly(phi)
        
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
        phi, dphi = np.linspace(2*np.pi,0,nphi,endpoint=False,retstep=True)
        if (self.tangent_poly is None):
            self.update_solutions()
            
        M = self.tangent_poly(phi)
        lam = self.adjoint_tangent_poly(phi)

        d_m_by_dcoilcurrents = self.compute_d_m_dcoilcurrents(phi)
            
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
            self.stellarator.reduce_current_derivatives([ires for ires in 
                                                         d_iota_dcoilcurrents])
    
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
        if (self.tangent_poly is None):
            self.update_solutions()
            
        M = self.tangent_poly(phi)
        lam = self.adjoint_tangent_poly(phi)
        iota = self.compute_iota()
        d_m_by_dcoilcoeffs = self.compute_d_m_dcoilcoeffs(phi)

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
    
    def compute_adjoint_tangent(self,phi,axis_poly=None):
        """
        For biotsavart and magnetic_axis objects, compute adjoint variable 
            by solving initial value probme. 
                
        Inputs:
            phi (1d array): toroidal angle for evaluation of adjoint variable
        """
        t_span = (2*np.pi,0)
        y0 = np.array([1,0,0,1])
        if self.constrained:
            args = (axis_poly,)
        else:
            args = ()
        out = scipy.integrate.solve_ivp(self.adjoint_rhs_fun,t_span,y0,
                                vectorized=False,rtol=self.rtol,atol=self.atol,
                                       t_eval=phi,args=args,dense_output=True)
        if (out.status==0):
            return out.y, out.sol
        else:
            raise RuntimeError('Error ocurred in integration of adjoint tangent map.')
            
    def adjoint_rhs_fun(self,phi,M,axis_poly=None):
        """
        Computes the RHS of the adjoint tangent map ODE, e.g. lambda'(phi) = rhs
            for given phi and lambda

        Inputs:
            phi (double): toroidal angle for evaluation
            lambda (1d array (4)): current value of adjoint tangent map
        Outputs:
            rhs (1d array (4)): rhs of ode
        """
        if axis_poly is not None:
            m = self.compute_m(phi,axis_poly(phi))
        else:
            m = self.compute_m(phi)

        return -np.squeeze(np.array([m[0]*M[0] + m[2]*M[2], m[0]*M[1] + m[2]*M[3], 
                                     m[1]*M[0] + m[3]*M[2], m[1]*M[1] + m[3]*M[3]]))

    def compute_d_m_dcoilcoeffs(self,phi):
        """
        Computes the derivative of matrix that appears on the rhs of the tangent map ode, 
        e.g. M'(phi) = m(phi) rhs, with respect to coil coeffs for given phi. 

        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_m_dcoilcoeffs (list (ncoils) of 3d array (npoints,4,ncoeffs)): derivative 
                of matrix appearing on rhs on ode wrt coil coeffs
        """
        if self.constrained:
            axis = self.axis_poly(phi)
            gamma = np.zeros((len(phi),3))
            gamma[...,0] = axis[0,...]*np.cos(phi)
            gamma[...,1] = axis[0,...]*np.sin(phi)
            gamma[...,2] = axis[1,...]
            self.biotsavart.set_points(gamma) 
        else:
            points = phi/(2*np.pi)
            self.magnetic_axis.points = np.asarray(points)
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
        if self.constrained:
            axis = self.axis_poly(phi)
            gamma = np.zeros((len(phi),3))
            gamma[...,0] = axis[0,...]*np.cos(phi)
            gamma[...,1] = axis[0,...]*np.sin(phi)
            gamma[...,2] = axis[1,...]
            self.biotsavart.set_points(gamma) 
        else:
            points = phi/(2*np.pi)
            self.magnetic_axis.points = np.asarray(points)
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
    
    def compute_d_m_d_magneticaxiscoeffs(self,phi):
        """
        Computes the derivative of  matrix that appears on the rhs of the tangent map ode, 
            e.g. M'(phi) = m(phi) M(phi), with respect to axis coefficients for given phi. 

        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_m_dcoilcoeffs (3d array (npoints,ncoeffs,4)): derivative 
                of matrix appearing on rhs on ode wrt axis coeffs
        """
        points = phi/(2*np.pi)
        self.magnetic_axis.points = np.asarray(points)
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
    
    def compute_grad_m(self,phi,poly):
        """
        Computes the derivative of  matrix that appears on the rhs of the 
            tangent map ode, e.g. M'(phi) = m(phi) M(phi), with respect, to 
            cylindrical R and Z.

        Inputs:
            phi (1d array): toroidal angles for evaluation
            poly : polynomial representation of axis solution
        Outputs:
            d_m_d_R 
            d_m_d_Z
        """
        axis = poly(phi)
        gamma = np.zeros((len(phi),3))
        gamma[:,0] = axis[0,:]*np.cos(phi)
        gamma[:,1] = axis[0,:]*np.sin(phi)
        gamma[:,2] = axis[1,:]
        self.biotsavart.set_points(gamma)
        
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
        dRdX = X/R
        dRdY = Y/R
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        dBRdR = (X**2*dBXdX + X*Y * (dBYdX + dBXdY) + Y**2 * dBYdY)/(R**2)
        dBPdR = (X*Y * (dBYdY-dBXdX) + X**2 * dBYdX - Y**2 * dBXdY)/(R**2)
        dBZdR =  dBZdX*X/R + dBZdY*Y/R
        dBRdZ =  dBXdZ*X/R + dBYdZ*Y/R
        dBPdZ = -dBXdZ*Y/R + dBYdZ*X/R 
               
        # Shape: ((len(points), 3, 3, 3))
        d2Bbs_by_dXdX = self.biotsavart.d2B_by_dXdX
        d2BXdX2 = d2Bbs_by_dXdX[...,0,0,0]
        d2BYdX2 = d2Bbs_by_dXdX[...,0,0,1]
        d2BZdX2 = d2Bbs_by_dXdX[...,0,0,2]
        d2BXdY2 = d2Bbs_by_dXdX[...,1,1,0]
        d2BYdY2 = d2Bbs_by_dXdX[...,1,1,1]
        d2BZdY2 = d2Bbs_by_dXdX[...,1,1,2]
        d2BXdZ2 = d2Bbs_by_dXdX[...,2,2,0]
        d2BYdZ2 = d2Bbs_by_dXdX[...,2,2,1]
        d2BZdZ2 = d2Bbs_by_dXdX[...,2,2,2]
        d2BXdXdY = d2Bbs_by_dXdX[...,0,1,0]
        d2BYdXdY = d2Bbs_by_dXdX[...,0,1,1]
        d2BZdXdY = d2Bbs_by_dXdX[...,0,1,2]
        d2BXdXdZ = d2Bbs_by_dXdX[...,0,2,0]
        d2BYdXdZ = d2Bbs_by_dXdX[...,0,2,1]
        d2BZdXdZ = d2Bbs_by_dXdX[...,0,2,2]
        d2BXdYdZ = d2Bbs_by_dXdX[...,1,2,0]
        d2BYdYdZ = d2Bbs_by_dXdX[...,1,2,1]
        d2BZdYdZ = d2Bbs_by_dXdX[...,1,2,2]
        
        d2BRdR2 = ((2 * X * dBXdX + X**2 * d2BXdX2 + Y * (dBYdX  + dBXdY) 
            + X * Y * (d2BYdX2 + d2BXdXdY)  + Y**2 * d2BYdXdY) * np.cos(phi) \
            + (X**2 * d2BXdXdY + X * (dBYdX + dBXdY) + X * Y * (d2BYdXdY + d2BXdY2) 
            + 2 * Y * dBYdY + Y**2 * d2BYdY2)*np.sin(phi))/(R**2) \
            - 2*dBRdR/R
        d2BRdRdZ = (X**2*d2BXdXdZ + X*Y * (d2BYdXdZ + d2BXdYdZ) 
                    + Y**2 * d2BYdYdZ)/(R**2)
#         dBPdR = (X*Y * (dBYdY-dBXdX) + X**2 * dBYdX - Y**2 * dBXdY)/(R**2)
        d2BPdR2 = ((Y * (dBYdY-dBXdX) + X * Y * (d2BYdXdY - d2BXdX2) 
                  + 2 * X * dBYdX + X**2 * d2BYdX2 - Y**2 * d2BXdXdY) * np.cos(phi) \
            + (X * (dBYdY-dBXdX) + X*Y* (d2BYdY2-d2BXdXdY) + X**2 * d2BYdXdY
              - 2 * Y * dBXdY - Y**2 * d2BXdY2) * np.sin(phi))/(R**2) \
            - 2*dBPdR/R
        d2BPdRdZ = (X*Y * (d2BYdYdZ-d2BXdXdZ) + X**2 * d2BYdXdZ 
                    - Y**2 * d2BXdYdZ)/(R**2)
#         dBZdR =  dBZdX*X/R + dBZdY*Y/R
        d2BZdR2 = ((dBZdX + X * d2BZdX2 + Y * d2BZdXdY) * np.cos(phi) 
                 + (X * d2BZdXdY + dBZdY + Y * d2BZdY2) * np.sin(phi))/R \
            - dBZdR/R
        d2BZdRdZ = (d2BZdXdZ*X + d2BZdYdZ*Y)/R
#         dBRdZ =  dBXdZ*X/R + dBYdZ*Y/R
        d2BRdRdZ = ((dBXdZ + X * d2BXdXdZ + Y * d2BYdXdZ) * np.cos(phi)
                + (X * d2BXdYdZ + dBYdZ + Y * d2BYdYdZ) * np.sin(phi))/R \
            - dBRdZ/R
        d2BRdZ2 = (d2BXdZ2*X + d2BYdZ2*Y)/R
#         dBPdZ = -dBXdZ*Y/R + dBYdZ*X/R 
        d2BPdRdZ = ((-Y * d2BXdXdZ + dBYdZ + X * d2BYdXdZ)*np.cos(phi)
                  + (- dBXdZ + Y * d2BXdYdZ + X * d2BYdYdZ)*np.sin(phi))/R \
            - dBPdZ/R
        d2BPdZ2 = (-d2BXdZ2*Y + d2BYdZ2*X)/R 
        
        dmdR = np.zeros((4,len(R)))
        dmdZ = np.zeros((4,len(Z)))
#       m[...,0] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
        dmdR[0,...] = dBRdR/BP - BR*dBPdR/BP**2 \
            + R*(d2BRdR2/BP - dBRdR*dBPdR/BP**2 - dBRdR*dBPdR/BP**2
                - BR*d2BPdR2/BP**2 + 2*BR*dBPdR*dBPdR/BP**3) \
            + (dBRdR/BP - BR*dBPdR/BP**2)
        dmdZ[0,...] = dBRdZ/BP - BR*dBPdZ/BP**2 \
            + R*(d2BRdRdZ/BP - dBRdR*dBPdZ/BP**2 - dBRdZ*dBPdZ/BP**2
                - BR*d2BPdRdZ/BP**2 + 2*BR*dBPdR*dBPdZ/BP**3) 
#       m[...,1] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
        dmdR[1,...] = R*(d2BRdRdZ/BP - dBRdZ*dBPdR/BP**2 - dBRdR*dBPdZ/BP**2
                - BR*dBPdRdZ/BP**2 + 2*BR*dBPdZ*dBPdR/BP**3) \
                + (dBRdZ/BP - BR*dBPdZ/BP**2)
        dmdZ[1,...] = R*(d2BRdZ2/Bp - dBRdZ*dBPdZ/BP**2 - dBRdZ*dBPdZ/BP**2
                - BR*d2BPdZ2/BP**2 + 2*BR*dBPdZ*dBPdZ/BP**3)
#       m[...,2] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
        dmdR[2,...] = dBZdR/BP - BZ*dBPdR/BP**2 \
            + R*(d2BZdR2/BP - dBZdR*dBPdR/BP**2 - dBZdR*dBPdR/BP**2
                - BZ*d2BPdR2/BP**2 + 2*BZ*dBPdR*dBPdR/(BP**3)) \
                + (dBZdR/BP - BZ*dBPdR/BP**2)
        dmdZ[2,...] = dBZdZ/Bp - BZ*dBPdZ/BP**2 \
            + R*(d2BZdRdZ/BP - dBPdZ*dBZdR/BP**2 - dBZdZ*dBPdR/BP**2
                - BZ*d2BPdRdZ/BP**2 + 2 * BZ * dBPdR * dBPdZ/BP**3)
#       m[...,3] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
        dmdR[3,...] = R*(d2BZdRdZ/BP - dBZdZ*dBPdR/BP**2 
                - dBZdR*dBPdZ/BP**2 - BZ*d2BPdRdZ/BP**2 + 2*BZ*dBPdZ*dBPdR/BP**3) \
                + (dBZdZ/BP - BZ*dBPdZ/BP**2)
        dmdZ[3,...] = R*(d2BZdZ2/Bp - dBZdZ*dBPdZ/BP**2
                - dBZdZ*dBPdZ/BP**2 - BZ*d2BPdZ2/BP**2 + 2*BZ*dBPdZ*dBPdZ/BP**3)
            
        return dmdR, dmdZ
    
    def res_axis(self,nphi=100):
        """
        Computes the residual between parameterization axis and "true" magnetic
        axis
        
        Inputs:
            nphi (int): number of gripdoints for evaluation of integral
            tol (double): tolerance for axis solve
        Outputs:
            res_axis (double): residual between parameterization axis
                and true axis 
        """
        phi, dphi = np.linspace(0,2*np.pi,nphi,endpoint=False,retstep=True)
        if self.axis_poly is None:
            self.update_solutions()
        axis = self.axis_poly(phi)
        
        self.magnetic_axis.points = np.asarray(phi/(2*np.pi))
        self.magnetic_axis.update()
        Rma = np.sqrt(self.magnetic_axis.gamma[:,0]**2 + self.magnetic_axis.gamma[:,1]**2)
        Zma = self.magnetic_axis.gamma[:,2]
        return 0.5*np.sum((axis[0,:]-Rma)**2 + (axis[1,:]-Zma)**2)*dphi

    def d_res_axis_d_magneticaxiscoeffs(self,nphi=100):
        """
        Compute derivative of res_axis wrt axis coefficients
        
        Inputs:
            nphi (int): number of gripdoints for evaluation of integral
            tol (double): tolerance for axis solve
        Outputs:
            d_res_axis_d_magneticaxiscoeffs (1d array (ncoeffs)): derivative of 
                residual between parameterization axis and true axis wrt axis 
                coeffs
        """
        phi, dphi = np.linspace(0,2*np.pi,nphi,endpoint=False,retstep=True)
        
        if self.axis_poly is None:
            self.update_solutions()
        axis = self.axis_poly(phi)
        
        self.magnetic_axis.points = np.asarray(phi/(2*np.pi))
        self.magnetic_axis.update()
        
        Rma = np.sqrt(self.magnetic_axis.gamma[:,0]**2 + self.magnetic_axis.gamma[:,1]**2)
        Zma = self.magnetic_axis.gamma[:,2]
        d_Rma = (self.magnetic_axis.dgamma_by_dcoeff[...,0]*self.magnetic_axis.gamma[:,0,None] + 
                 self.magnetic_axis.dgamma_by_dcoeff[...,1]*self.magnetic_axis.gamma[:,1,None]) \
            / Rma[:,None]
        d_Zma = self.magnetic_axis.dgamma_by_dcoeff[...,2]
        
        return np.sum((Rma[:,None]-axis[0,:,None])*d_Rma 
                    + (Zma[:,None]-axis[1,:,None])*d_Zma,axis=0)*dphi
        
    def d_res_axis_d_coil_currents(self,nphi=100):
        """
        Compute derivative of res_axis wrt coil currents
        
        Inputs:
            nphi (int): number of gripdoints for evaluation of integral
            tol (double): tolerance for axis solve
        Outputs:
            d_res_axis_d_coil_currents (list of doubles): derivatives of 
                residual between parameterization axis and true axis wrt coil
                currents
        """
        phi,dphi = np.linspace(0,2*np.pi,nphi,endpoint=False,retstep=True)
        
        if (self.axis_poly is None or self.adjoint_axis_poly is None):
            self.update_solutions()
        axis = self.axis_poly(phi)
        mu = self.adjoint_axis_poly(phi)

        d_V_by_dcoilcurrents = self.compute_d_V_dcoilcurrents(phi,self.axis_poly)
        d_res_axis_dcoilcurrents = []
        for i in range(len(d_V_by_dcoilcurrents)):
            d_V = d_V_by_dcoilcurrents[i]
            mu_dot_d_V = mu[0,:]*d_V[0,...] + mu[1,:]*d_V[1,:]
            d_res_axis = - np.sum(mu_dot_d_V)*dphi
            d_res_axis_dcoilcurrents.append(d_res_axis)
        d_res_axis_dcoilcurrents = \
            self.stellarator.reduce_current_derivatives([ires for ires in d_res_axis_dcoilcurrents])
    
        return d_res_axis_dcoilcurrents
    
    def d_res_axis_d_coil_coeffs(self,nphi=100):
        """
        Compute derivative of res_axis wrt coil coefficients
        
        Inputs:
            nphi (int): number of gripdoints for evaluation of integral
            tol (double): tolerance for axis solve
        Outputs:
            d_res_axis_d_coil_currents (list of 1d arrays (ncoeffs)): derivatives of 
                residual between parameterization axis and true axis wrt coil
                coeffs
        """
        phi,dphi = np.linspace(0,2*np.pi,nphi,endpoint=False,retstep=True)    
        
        if (self.axis_poly is None or self.adjoint_axis_poly is None):
            self.update_solutions()
        axis = self.axis_poly(phi)
        mu = self.adjoint_axis_poly(phi)
        
        d_V_by_dcoilcoeffs = self.compute_d_V_dcoilcoeffs(phi,self.axis_poly)
        d_res_axis_dcoilcoeffs = []
        for i in range(len(d_V_by_dcoilcoeffs)):
            d_V = d_V_by_dcoilcoeffs[i]
            mu_dot_d_V = mu[0,...,None]*d_V[0,...] + mu[1,...,None]*d_V[1,...]
            d_res_axis = - np.sum(mu_dot_d_V,axis=0)*dphi
            d_res_axis_dcoilcoeffs.append(d_res_axis)
        d_res_axis_dcoilcoeffs = self.stellarator.reduce_coefficient_derivatives([ires for ires in d_res_axis_dcoilcoeffs])
        return d_res_axis_dcoilcoeffs

    def compute_adjoint_axis(self,phi,axis_poly,tangent_poly=None,adjoint_tangent_poly=None):
        """
        Computes adjoint variable required for computing derivative of 
            axis_res metric
            
        Inputs:
            phi (1d array): toroidal angle for evaluation of adjoint variable
            poly (instance of scipy.interpolate.PPoly cubic spline): polyomial
                representing magnetic axis solution
            adjoint_tangent_poly (instance of scipy.interpolate.PPoly cubic spline): polyomial
                representing adjoint variable for tangent map
        """
        y0 = axis_poly(phi)
        if (self.constrained):
            fun = lambda x,y : self.rhs_fun_adjoint(x,y,axis_poly,tangent_poly,
                                                    adjoint_tangent_poly)
        else:
            fun = lambda x,y : self.rhs_fun_adjoint(x,y,axis_poly)
        fun_jac = lambda x,y : self.jac_adjoint(x,y,axis_poly)
        out = scipy.integrate.solve_bvp(fun=fun,
                                        bc=self.bc_fun_axis,
                                        x=phi,y=y0,fun_jac=fun_jac,
                                        bc_jac=self.bc_jac,verbose=self.verbose,
                                        tol=self.tol,max_nodes=self.max_nodes)
        if (out.status==0):
            # Evaluate polynomial on grid
            return out.sol(phi), out.sol
        else:
            raise RuntimeError('Error ocurred in integration of axis.')

    def compute_axis(self,phi):
        """
        For biotsavart and magnetic_axis objects, compute rotational transform
            from tangent map by solving initial value problem.
        
        Inputs:
            phi (1d array): 1d array for evaluation of tangent map
        Outputs:
            y (2d array (2,len(phi))): axis on grid of toroidal angle
        """
        if (self.axis_poly is not None):
            y0 = self.axis_poly(phi)
        else:
            self.magnetic_axis.points = np.asarray(phi/(2*np.pi))
            self.magnetic_axis.update()
            axis = self.magnetic_axis.gamma
            y0 = np.zeros((2,len(phi)))
            y0[0,:] = np.sqrt(axis[:,0]**2 + axis[:,1]**2)
            y0[1,:] = axis[:,2]
        
        out = scipy.integrate.solve_bvp(fun=self.rhs_fun_axis,bc=self.bc_fun_axis,
                                        x=phi,y=y0,fun_jac=self.compute_jac,
                                        bc_jac=self.bc_jac,verbose=self.verbose,
                                        tol=self.tol,max_nodes=self.max_nodes)
        if (out.status==0):
            # Evaluate polynomial on grid
            return out.sol(phi), out.sol
        else:
            raise RuntimeError('Error ocurred in integration of axis.')
        
    def rhs_fun_axis(self,phi,axis):
        """
        Computes rhs of magnetic field line flow ode, i.e.
            r'(\phi) = V(\phi)
            
        Inputs:
            phi (1d array): toroidal angle for evaluation of rhs
            axis (2d array (2,len(phi))): R and Z for evaluation of rhs
        Outputs:
            V (2d array (2,len(phi))): R and Z components of rhs
        """
        gamma = np.zeros((len(phi),3))
        gamma[:,0] = axis[0,:]*np.cos(phi)
        gamma[:,1] = axis[0,:]*np.sin(phi)
        gamma[:,2] = axis[1,:]
        self.biotsavart.set_points(gamma)

        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]
        X = self.biotsavart.points[:,0]
        Y = self.biotsavart.points[:,1]
        Z = self.biotsavart.points[:,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        V = np.zeros((2,len(R)))
        V[0,:] = R*BR/BP
        V[1,:] = R*BZ/BP
        return V
    
    def rhs_fun_adjoint(self,phi,eta,axis_poly,tangent_poly=None,adjoint_poly=None):
        """
        Compute rhs of adjoint problem for res_axis metric, i.e.
            \mu'(\phi) = V(\phi)
            
        Inputs:
            phi (1d array): toroidal angle for evaluation of rhs
            eta (2d array (2,len(phi))): mu_R and mu_Z for evaluation of rhs
            axis_poly (instance of scipy.interpolate.PPoly cubic spline): polynomial
                representing magnetic axis solution
            tangent_poly (instance of scipy.interpolate.PPoly cubic spline): polynomial
                representing tangent map solution
            adjoint_poly (instance of scipy.interpolate.PPoly cubic spline): polynomial
                representing "lambda" adjoint solution 
        Outputs:
            V (2d array (2,len(phi))): R and Z components of rhs
        """
        if (self.constrained):
            M = tangent_poly(phi)
            lam = adjoint_poly(phi)
            dmdR, dmdZ = self.compute_grad_m(phi,axis_poly)
            lambda_dot_dmdR_times_M = \
                  lam[0,:,None]*(dmdR[0,...]*M[0,...,None] + dmdR[1,...]*M[2,...,None]) \
                + lam[1,:,None]*(dmdR[0,...]*M[1,...,None] + dmdR[1,...]*M[3,...,None]) \
                + lam[2,:,None]*(dmdR[2,...]*M[0,...,None] + dmdR[3,...]*M[2,...,None]) \
                + lam[3,:,None]*(dmdR[2,...]*M[1,...,None] + dmdR[3,...]*M[3,...,None])
            lambda_dot_dmdZ_times_M = \
                  lam[0,:,None]*(dmdZ[0,...]*M[0,...,None] + dmdZ[1,...]*M[2,...,None]) \
                + lam[1,:,None]*(dmdZ[0,...]*M[1,...,None] + dmdZ[1,...]*M[3,...,None]) \
                + lam[2,:,None]*(dmdZ[2,...]*M[0,...,None] + dmdZ[3,...]*M[2,...,None]) \
                + lam[3,:,None]*(dmdZ[2,...]*M[1,...,None] + dmdZ[3,...]*M[3,...,None])
        else:
            self.magnetic_axis.points = np.asarray(phi/(2*np.pi))
            self.magnetic_axis.update()
            gamma_ma = self.magnetic_axis.gamma
            R_ma = np.sqrt(gamma_ma[:,0]**2 + gamma_ma[:,1]**2)
            Z_ma = gamma_ma[:,2]
        
        axis = axis_poly(phi)
        gamma = np.zeros((len(phi),3))
        gamma[:,0] = axis[0,:]*np.cos(phi)
        gamma[:,1] = axis[0,:]*np.sin(phi)
        gamma[:,2] = axis[1,:]
        self.biotsavart.set_points(gamma)

        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]
        X = self.biotsavart.points[:,0]
        Y = self.biotsavart.points[:,1]
        Z = self.biotsavart.points[:,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        m = self.compute_m(phi)
        V = np.zeros((2,len(R)))
        if (self.constrained):
            V[0,:] = -m[0,:]*eta[0,:] - m[2,:]*eta[1,:] - lambda_dot_dmdR_times_M 
            V[1,:] = -m[3,:]*eta[1,:] - m[1,:]*eta[0,:] - lambda_dot_dmdZ_times_M           
        else:
            V[0,:] = -m[0,:]*eta[0,:] - m[2,:]*eta[1,:] + axis[0,:] - R_ma
            V[1,:] = -m[3,:]*eta[1,:] - m[1,:]*eta[0,:] + axis[1,:] - Z_ma
        return V
    
    def jac_adjoint(self,phi,y,axis_poly):
        """
        Computes jacobian of rhs of adjoint equation (rhs_fun_adjoint)
        
        Inputs:
            phi (1d array): toroidal angle for evaluation
            y (2d array (2,len(phi))): mu_R and mu_Z for evaluation
        Outputs:
            jac (3d array (2,2,len(phi))): jacobian matrix on phi grid
        """
        m = self.compute_m(phi,axis_poly(phi))
        jac = np.zeros((2,2,len(phi)))
        jac[0,0,:] = m[0,:]
        jac[1,0,:] = m[1,:]
        jac[0,1,:] = m[2,:]
        jac[1,1,:] = m[3,:]
        return -jac
    
    def compute_jac(self,phi,y):
        """
        Computes jacobian matrix for magnetic axis bvp (compute_axis) 
        
        Inputs:
            phi (1d array): toroidal angle for evaluation
            y (2d array (2,len(phi))): R and Z for evaluation
        Outputs:
            jac (3d array (2,2,len(phi))): jacobian matrix on phi grid
        """
        m = self.compute_m(phi,y)
        jac = np.zeros((2,2,len(phi)))
        jac[0,0,...] = m[0,...]
        jac[0,1,...] = m[1,...]
        jac[1,0,...] = m[2,...]
        jac[1,1,...] = m[3,...]
        return jac
    
    def bc_jac(self,ya,yb):
        """
        Jacobian for boundary condition function for magnetic axis bvp (bc_fun_axis)
        
        Inputs:
            ya (1d array(2)): axis solution at phi = 0
            yb (1d array(2)): axis solution at phi = 2\pi
        Outputs:
            jac (2d array(2,2), 2d array(2,2)): jacobian for boundary condition 
                at 0 and 2\pi
        """
        return np.eye(2), -np.eye(2)
        
    def bc_fun_axis(self,axisa,axisb):
        """
        Boundary condition function for magnetic axis bvp (compute_axis)
        
        Inputs:
            axisa (1d array(2)): Magnetic axis solution at phi = 0
            axisb (1d array(2)): Magnetic axis solution at phi = 2*pi
        """
        return axisa - axisb
    
    def compute_d_V_dcoilcurrents(self,phi,axis_poly):
        """
        Computes the derivative of the rhs of the 
            axis ode, e.g. r'(phi) = V(phi), with respect to coil coeffs for 
            given phi. 

        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_V_dcoilcurrents (list (ncoils) of 2d array (2,npoints)): derivative 
                of V wrt coil currents
        """
        axis = axis_poly(phi)
        gamma = np.zeros((len(phi),3))
        gamma[:,0] = axis[0,:]*np.cos(phi)
        gamma[:,1] = axis[0,:]*np.sin(phi)
        gamma[:,2] = axis[1,:]
        self.biotsavart.set_points(gamma)
        
        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]
        
        X = self.biotsavart.points[:,0]
        Y = self.biotsavart.points[:,1]
        Z = self.biotsavart.points[:,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        
        # Shape: (ncoils,npoints,3)
        dB_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents
        
        d_V_by_dcoilcurrents = []
        for i in range(len(dB_by_dcoilcurrents)):
            d_B = dB_by_dcoilcurrents[i]
            d_BX = d_B[...,0]
            d_BY = d_B[...,1]
            d_BZ = d_B[...,2]
                
            d_BR =  (X * d_BX + Y * d_BY)/R
            d_BP = (-Y * d_BX + X * d_BY)/R
        
            d_V = np.zeros((2,np.shape(d_BR)[0]))
            d_V[0,...] = R * (d_BR/BP - BR*d_BP/BP**2)
            d_V[1,...] = R * (d_BZ/BP - BZ*d_BP/BP**2)
            d_V_by_dcoilcurrents.append(d_V)
            
        return d_V_by_dcoilcurrents
    
    def compute_d_V_dcoilcoeffs(self,phi,axis_poly):
        """
        Computes the derivative of the rhs of the 
            axis ode, e.g. r'(phi) = V(phi), with respect to coil coeffs for 
            given phi. 

        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_V_dcoilcoeffs (list (ncoils) of 3d array (2,npoints,ncoeffs)): 
                derivative of V wrt coil coeffs
        """
        axis = axis_poly(phi)
        gamma = np.zeros((len(phi),3))
        gamma[:,0] = axis[0,:]*np.cos(phi)
        gamma[:,1] = axis[0,:]*np.sin(phi)
        gamma[:,2] = axis[1,:]
        self.biotsavart.set_points(gamma)
        
        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]
        
        X = self.biotsavart.points[:,0]
        Y = self.biotsavart.points[:,1]
        Z = self.biotsavart.points[:,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
       
        # Shape: (ncoils,npoints,nparams,3)
        dB_by_dcoilcoeffs = self.biotsavart.dB_by_dcoilcoeffs
        
        d_V_by_dcoilcoeffs = []
        for i in range(len(dB_by_dcoilcoeffs)):
            d_B = dB_by_dcoilcoeffs[i]
            d_BX = d_B[...,0]
            d_BY = d_B[...,1]
            d_BZ = d_B[...,2]
                
            d_BR =  (X[:,None] * d_BX + Y[:,None] * d_BY)/R[:,None]
            d_BP = (-Y[:,None] * d_BX + X[:,None] * d_BY)/R[:,None]
                    
            d_V = np.zeros((2,np.shape(d_BR)[0],np.shape(d_BR)[1]))
            d_V[0,...] = R[:,None] * (d_BR/BP[:,None] - BR[:,None]*d_BP/BP[:,None]**2)
            d_V[1,...] = R[:,None] * (d_BZ/BP[:,None] - BZ[:,None]*d_BP/BP[:,None]**2)
            d_V_by_dcoilcoeffs.append(d_V)
            
        return d_V_by_dcoilcoeffs

