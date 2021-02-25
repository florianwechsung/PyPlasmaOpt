import numpy as np
from math import pi
from grad_optimizer import GradOptimizer
from .biotsavart import BiotSavart

class QfmSurface():
    
    def __init__(self, mmax, nmax, nfp, stellarator, ntheta, nphi, volume):
        self.mmax = mmax
        self.nmax = nmax
        self.nfp = nfp
        self.biotsavart = BiotSavart(stellarator.coils, stellarator.currents)
        self.stellarator = stellarator
        self.mnmax,self.xm,self.xn = self.init_modes(mmax,nmax)
        self.xn = self.xn*nfp
        self.ntheta = ntheta
        self.nphi = nphi
        self.thetas,self.phis,self.dtheta,self.dphi = self.init_grid(ntheta,nphi)
        self.volume = volume
        
    def init_modes(self,mmax,nmax):
        """
        Initialize poloidal and toroidal mode number
        
        Inputs:
            mmax (int) : maximum poloidal mode number
            nmax (int) : maximum toroidal mode number
            
        Outputs: 
            mnmax (int) : number of moder numbers
            xm (int array) : poloidal mode numbers (1D array of length mnmax)
            xn (int array) : toroidal mode numbers (1D array of length mnmax)
        """
        mnmax = (nmax+1) + (2*nmax+1)*mmax
        xm = np.zeros(mnmax)
        xn = np.zeros(mnmax)
        # m = 0 modes
        ind = 0
        for jn in range(nmax+1):
            xn[ind] = jn
            ind += 1

        # m /= 0 modes
        for jm in range(1,mmax+1):
            for jn in range(-nmax,nmax+1):
                xm[ind] = jm
                xn[ind] = jn
                ind += 1

        return mnmax, xm, xn
    
    def params_full(self, params):
        """
        Compute full set of parameters (including R00) from prescribed set
        
        Inputs: 
            params (1d array of length 2*mnmax-1): surface Fourier parameters 
                excluding R00
        Outputs:
            params (1d array of length 2*mnmax): surface Fourier parameters 
                including R00
        """
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')

        [dRdtheta, dRdphi, dZdtheta, dZdphi] = self.position_derivatives(params)
        
        params = np.concatenate(([0],params),axis=0)
        paramsR = params[0:self.mnmax]
        paramsZ = params[self.mnmax::]

        nax = np.newaxis
        xm = self.xm[:,nax,nax]
        xn = self.xn[:,nax,nax]
        thetas = self.thetas[nax,:,:]
        phis = self.phis[nax,:,:]
        
        paramsR = paramsR[:,nax,nax]
        
        cos_angle = np.cos(xm * thetas - xn * phis)
        B = -0.5*np.sum(paramsR * cos_angle * dZdtheta[nax,:,:])*self.dtheta*self.dphi*self.nfp
        C = -0.5*np.sum(np.sum(paramsR * cos_angle,axis=0)**2 * dZdtheta)*self.dtheta*self.dphi*self.nfp
        R00 = (self.volume - C)/(2*B)
        params[0] = np.abs(R00)
        return params
    
    def d_params_full(self, params):
        """
        Compute derivative of params_full with resepct to params
        
        Inputs: 
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            d_params_full (2d array (2*mnmax, 2*mnmax-1)): derivatives
                of params_full with respect to params
        """
        [d_dRdtheta, d_dRdphi, d_dZdtheta, d_dZdphi] = self.d_position_derivatives(params)
        d_dZdtheta = d_dZdtheta[self.mnmax-1::]
        [dRdtheta, dRdphi, dZdtheta, dZdphi] = self.position_derivatives(params)

        nax = np.newaxis
        xm = self.xm[:,nax,nax]
        xn = self.xn[:,nax,nax]
        thetas = self.thetas[nax,:,:]
        phis = self.phis[nax,:,:]
        cos_angle = np.cos(xm * thetas - xn * phis)
        
        params = np.concatenate(([0],params))
        paramsR = params[0:self.mnmax]
        paramsR = paramsR[:,nax,nax]
                
        B = -0.5*np.sum(paramsR * cos_angle * dZdtheta[nax,:,:]) * self.dtheta \
            *self.dphi*self.nfp
        d_B_R = -0.5*np.sum(cos_angle * dZdtheta[nax,:,:],axis=(1,2))  \
            *self.dtheta*self.dphi*self.nfp
        d_B_Z = -0.5*np.sum(np.sum(paramsR * cos_angle, axis=0)[nax,:,:] 
            * d_dZdtheta,axis=(1,2))*self.dtheta*self.dphi*self.nfp
        C = -0.5*np.sum(np.sum(paramsR * cos_angle, axis=0)**2 * dZdtheta) \
            * self.dtheta*self.dphi*self.nfp
        d_C_Z = -0.5*np.sum(np.sum(paramsR * cos_angle, axis=0)[nax,:,:]**2 
            * d_dZdtheta,axis=(1,2))*self.dtheta*self.dphi*self.nfp
        d_C_R = -0.5*np.sum(2 * np.sum(paramsR * cos_angle, axis=0)[nax,:,:] 
            * cos_angle * dZdtheta[nax,:,:],axis=(1,2)) \
            * self.dtheta*self.dphi*self.nfp
        R00 = (self.volume - C)/(2*B)
        d_R00_Z = -d_C_Z/(2*B) - R00*d_B_Z/B
        d_R00_R = -d_C_R/(2*B) - R00*d_B_R/B
        d_R00 = np.hstack((d_R00_R,d_R00_Z))*np.sign(R00)
        eye = np.vstack((np.zeros(len(params)-1),np.eye(len(params)-1)))

        d_params_full = np.zeros((len(params),len(params)-1))
        d_params_full[0,:] = d_R00[1::]
        d_params_full += eye
        return d_params_full
    
    def init_grid(self,ntheta,nphi):
        """
        Compute regular grid in poloidal and toroidal angles (meshgridded)
        
        Inputs:
            ntheta (int): number of poloidal gridpoints per 2*pi period
            nphi (int): number of toroidal gridpoints per 2*pi/nfp period
            
        Outputs:
            thetas (2d array (nphi,ntheta)): poloidal angle grid
            phis (2d array (nphi,ntheta)): toroidal angle grid
            dtheta (float): poloidal angle grid spacing
            dphi (float): toroidal angle grid spacing
        """
        thetas = np.linspace(0,2*np.pi,self.ntheta,endpoint=False)
        phis = np.linspace(0,2*np.pi/self.nfp,self.nphi,endpoint=False)
        dtheta = thetas[1]-thetas[0]
        dphi = phis[1]-phis[0]
        [thetas,phis] = np.meshgrid(thetas,phis)
        return thetas, phis, dtheta, dphi
    
    def position(self,params):
        """
        Cylindrical coordinate (R,Z) position 
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            R (2d array (nphi,ntheta)): radius on angular grid
            Z (2d array (nphi,ntheta)): height on angular grid
        """
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')
        
        params = self.params_full(params)
        paramsR = params[0:self.mnmax]
        paramsZ = params[self.mnmax::]
            
        nax = np.newaxis
        xm = self.xm[:,nax,nax]
        xn = self.xn[:,nax,nax]
        thetas = self.thetas[nax,:,:]
        phis = self.phis[nax,:,:]
        paramsR = paramsR[:,nax,nax]
        paramsZ = paramsZ[:,nax,nax]
        angle = xm * thetas - xn * phis
        R = np.sum(paramsR * np.cos(angle),axis=0)
        Z = np.sum(paramsZ * np.sin(angle),axis=0)
        
        return R, Z
    
    def d_position(self,params):
        """
        Derivative of cylindrical coordinate (R,Z) position with respect
            to surface parameters 
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            d_R (3d array (len(params),nphi,ntheta)): derivative of radius on 
                angular grid wrt params
            d_Z (3d array (len(params),nphi,ntheta)): derivative of height on 
                angular grid wrt params
        """
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')
            
        d_params_full = self.d_params_full(params)
        params = self.params_full(params)
        paramsR = params[0:self.mnmax]
        paramsZ = params[self.mnmax::]
            
        nax = np.newaxis
        xm = self.xm[:,nax,nax]
        xn = self.xn[:,nax,nax]
        thetas = self.thetas[nax,:,:]
        phis = self.phis[nax,:,:]
        paramsR = paramsR[:,nax,nax]
        paramsZ = paramsZ[:,nax,nax]
        angle = xm * thetas - xn * phis
        d_R =  np.cos(angle)
        d_Z =  np.sin(angle)
        
        d_params_full_R = d_params_full[0:self.mnmax]
        d_params_full_Z = d_params_full[self.mnmax::]
                                                                    
        d_R = np.sum(d_R[:,nax,...] * d_params_full_R[:,:,nax,nax],axis=0)
        d_Z = np.sum(d_Z[:,nax,...] * d_params_full_Z[:,:,nax,nax],axis=0)
        return d_R, d_Z    
        
    def position_derivatives(self,params):
        """
        Computes derivatives of cylindrical coordinates wrt angles
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            dRdtheta (2d array (nphi,ntheta)): derivative of radius on angular 
                grid wrt poloidal angle
            dRdphi (2d array (nphi,ntheta)): derivative of radius on angular 
                grid wrt toroidal angle
            dZdtheta (2d array (nphi,ntheta)): derivative of height on angular 
                grid wrt poloidal angle
            dZdphi (2d array (nphi,ntheta)): derivative of height on angular 
                grid wrt toroidal angle
        """
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')
        
        # Position derivatives does not depend on R00
        params = np.concatenate(([0],params),axis=0)
        paramsR = params[0:self.mnmax]

        paramsZ = params[self.mnmax::]
            
        nax = np.newaxis
        xm = self.xm[:,nax,nax]
        xn = self.xn[:,nax,nax]
        thetas = self.thetas[nax,:,:]
        phis = self.phis[nax,:,:]
        paramsR = paramsR[:,nax,nax]
        paramsZ = paramsZ[:,nax,nax]
        angle = xm * thetas - xn * phis
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        dRdtheta = np.sum(- xm * paramsR * np.sin(angle),axis=0)
        dRdphi  = np.sum(  xn * paramsR * np.sin(angle),axis=0)
        dZdtheta = np.sum(  xm * paramsZ * np.cos(angle),axis=0)
        dZdphi  = np.sum(- xn * paramsZ * np.cos(angle),axis=0)
        
        return dRdtheta, dRdphi, dZdtheta, dZdphi
        
    def d_position_derivatives(self,params):
        """
        Derivative of angular derivatives of cylindrical coordinages with respect
            to surface parameters 
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            d_dRdtheta (3d array (len(params),nphi,ntheta)): derivative of 
                dRdtheta on angular grid wrt params
            d_dRdphi (3d array (len(params),nphi,ntheta)): derivative of 
                dRdphi on angular grid wrt params
            d_dZdtheta (3d array (len(params),nphi,ntheta)): derivative of 
                dZdtheta on angular grid wrt params 
            d_dZdphi (3d array (len(params),nphi,ntheta)): derivative of 
                dZdphi on angular grid wrt params        
        """
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')
        
        paramsR = params[0:self.mnmax]
        paramsZ = params[self.mnmax::]
            
        nax = np.newaxis
        xm = self.xm[:,nax,nax]
        xn = self.xn[:,nax,nax]
        thetas = self.thetas[nax,:,:]
        phis = self.phis[nax,:,:]
        paramsR = paramsR[:,nax,nax]
        paramsZ = paramsZ[:,nax,nax]
        angle = xm * thetas - xn * phis
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        d_dRdtheta = - xm * np.sin(angle)
        d_dRdphi  =   xn * np.sin(angle)
        d_dZdtheta =   xm * np.cos(angle)
        d_dZdphi  = - xn * np.cos(angle)
                              
        d_dRdtheta = np.vstack((d_dRdtheta,np.zeros_like(d_dZdtheta)))[1::,:]
        d_dRdphi = np.vstack((d_dRdphi,np.zeros_like(d_dZdtheta)))[1::,:]
        d_dZdtheta = np.vstack((np.zeros_like(d_dZdtheta),d_dZdtheta))[1::,:]
        d_dZdphi = np.vstack((np.zeros_like(d_dZdphi),d_dZdphi))[1::,:]

        return d_dRdtheta, d_dRdphi, d_dZdtheta, d_dZdphi
    
    def norm_normal(self,params):
        """
        Computes surface area Jacobian on angular grid from surface parameters
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            norm_normal (2d array (nphi,ntheta)): surface area Jacobian on 
                angular grid
        """
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')
    
        dRdtheta, dRdphi, dZdtheta, dZdphi = self.position_derivatives(params)
        R, Z = self.position(params)
        
        return np.sqrt(R**2 * (dRdtheta**2 + dZdtheta**2) 
                    + (dZdtheta*dRdphi - dZdphi*dRdtheta)**2)
    
    def d_norm_normal(self,params):
        """
        Computes derivatives of surface area Jacobian on angular grid wrt 
            surface parameters
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            d_norm_normal (3d array (len(params),nphi,ntheta)): derivative of 
                norm_normal on angular grid wrt params   
        """
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')
    
        dRdtheta, dRdphi, dZdtheta, dZdphi = self.position_derivatives(params)
        d_dRdtheta, d_dRdphi, d_dZdtheta, d_dZdphi = self.d_position_derivatives(params)
        R, Z = self.position(params)
        d_R, d_Z = self.d_position(params)
        N = self.norm_normal(params)
            
        nax = np.newaxis
        dRdtheta = dRdtheta[nax,:,:]
        dRdphi = dRdphi[nax,:,:]
        dZdtheta = dZdtheta[nax,:,:]
        dZdphi = dZdphi[nax,:,:]
        R = R[nax,:,:]        
        N = N[nax,:,:]
        
        d_N = (  R * d_R * (dRdtheta**2 + dZdtheta**2) 
               + R**2  * (dRdtheta*d_dRdtheta)
               + (dZdtheta*dRdphi   -  dZdphi*dRdtheta) 
               * (dZdtheta*d_dRdphi -  dZdphi*d_dRdtheta ))/N  \
            + ( R**2  * (dZdtheta*d_dZdtheta)
                  + (dZdtheta*dRdphi   - dZdphi*dRdtheta) 
                  * (d_dZdtheta*dRdphi -  d_dZdphi*dRdtheta))/N
        return d_N
            
    def normal(self,params):
        """
        Computes cylindrical componnets of unit normal vector on angular grid 
            from surface parameters
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            nR (2d array (nphi,ntheta)): R component of unit normal on angular
                grid
            nP (2d array (nphi,ntheta)): phi component of unit normal on angular
                grid
            nZ (2d array (nphi,ntheta)): Z component of unit normal on angular
                grid
        """
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')
    
        dRdtheta, dRdphi, dZdtheta, dZdphi = self.position_derivatives(params)
        R, Z = self.position(params)
        N    = self.norm_normal(params)
        
        NR = -dZdtheta * R
        NP =  dZdtheta * dRdphi - dZdphi * dRdtheta
        NZ =  dRdtheta * R
        
        return NR/N, NP/N, NZ/N
    
    def d_normal(self,params):
        """
        Computes derivatives of cylindrical component of unit normal on angular 
            grid wrt surface parameters
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            d_nR (3d array (len(params),nphi,ntheta)): derivative of 
                R component of unit normal on angular grid wrt params   
            d_nP (3d array (len(params),nphi,ntheta)): derivative of 
                phi component of unit normal on angular grid wrt params    
            d_nZ (3d array (len(params),nphi,ntheta)): derivative of 
                Z component of unit normal on angular grid wrt params    
        """
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')
        
        dRdtheta, dRdphi, dZdtheta, dZdphi = self.position_derivatives(params)
        d_dRdtheta, d_dRdphi, d_dZdtheta, d_dZdphi = self.d_position_derivatives(params)
        R, Z = self.position(params)
        d_R, d_Z = self.d_position(params)
        N = self.norm_normal(params)
        nR, nP, nZ = self.normal(params)
        d_N = self.d_norm_normal(params)
        
        nax = np.newaxis
        dRdtheta = dRdtheta[nax,:,:]
        dRdphi = dRdphi[nax,:,:]
        dZdtheta = dZdtheta[nax,:,:]
        dZdphi = dZdphi[nax,:,:]
        R = R[nax,:,:]
        nR = nR[nax,:,:]
        nP = nP[nax,:,:]
        nZ = nZ[nax,:,:]
        N = N[nax,:,:]
        
        d_NR = - dZdtheta * d_R - d_dZdtheta * R
        d_NP = dZdtheta * d_dRdphi - dZdphi * d_dRdtheta \
            + d_dZdtheta * dRdphi - d_dZdphi * dRdtheta
        d_NZ = d_dRdtheta * R + dRdtheta * d_R
        
        d_nR = d_NR/N - nR * d_N/N
        d_nZ = d_NZ/N - nZ * d_N/N
        d_nP = d_NP/N - nP * d_N/N
        
        return d_nR, d_nP, d_nZ
    
    def B_from_points(self,params):
        """
        Computes magnetic field on angular grid on surface computed from 
            provided parameters
            
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs;
            B (2d array (size(thetas), 3)): Cartesian components of magnetic
                field on flattened angular grid
        """
        R, Z = self.position(params)
        X = R * np.cos(self.phis)
        Y = R * np.sin(self.phis)
        
        points = np.zeros((len(X.flatten()), 3))
        points[:,0] = X.flatten()
        points[:,1] = Y.flatten()
        points[:,2] = Z.flatten()
        self.biotsavart.set_points(points)
        return self.biotsavart.B
    
    def d_B_from_points(self,params):
        """
        Computes derivative of magnetic field on angular grid on surface wrt 
            params

        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs;
            d_B (3d array (len(params), size(thetas), 3)): derivatives of 
                Cartesian components of magnetic field on flattened angular grid
                wrt params
        """
        nax = np.newaxis
        R, Z = self.position(params)
        X = R * np.cos(self.phis)
        Y = R * np.sin(self.phis)
        d_R, d_Z = self.d_position(params)
        d_X = d_R * np.cos(self.phis)
        d_Y = d_R * np.sin(self.phis)
        
        d_X = np.reshape(d_X,(len(params),len(R.flatten())))[...,nax]
        d_Y = np.reshape(d_Y,(len(params),len(R.flatten())))[...,nax]
        d_Z = np.reshape(d_Z,(len(params),len(R.flatten())))[...,nax]
        
        points = np.zeros((len(X.flatten()), 3))
        points[:,0] = X.flatten()
        points[:,1] = Y.flatten()
        points[:,2] = Z.flatten()
        self.biotsavart.set_points(points)
        gradB = self.biotsavart.dB_by_dX[nax,...]
        
        return d_X * gradB[...,0,:] + d_Y * gradB[...,1,:] + d_Z * gradB[...,2,:]
    
    def d_B_from_points_dcoilcoeff(self,params):
        R, Z = self.position(params)
        X = R * np.cos(self.phis)
        Y = R * np.sin(self.phis)
        
        points = np.zeros((len(X.flatten()), 3))
        points[:,0] = X.flatten()
        points[:,1] = Y.flatten()
        points[:,2] = Z.flatten()
        # Shape: (ncoils,npoints,nparams,3)
        return self.biotsavart.compute_by_dcoilcoeff(points).dB_by_dcoilcoeffs
    
    def d_B_from_points_dcoilcurrents(self,params):
        R, Z = self.position(params)
        X = R * np.cos(self.phis)
        Y = R * np.sin(self.phis)
        
        points = np.zeros((len(X.flatten()), 3))
        points[:,0] = X.flatten()
        points[:,1] = Y.flatten()
        points[:,2] = Z.flatten()
        # Shape: (ncoils,npoints,3)
        return self.biotsavart.compute(points).dB_by_dcoilcurrents

    def quadratic_flux(self,params):
        """
        Computes normalized quadratic flux integral:
            \int d^2 x (B \cdot n)^2/\int d^2 x B^2
            
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            quadratic_flux (float): normalized quadratic flux integral
        """
        B = self.B_from_points(params)
        
        N = self.norm_normal(params)
        nR, nP, nZ = self.normal(params)
        
        nX = (nR * np.cos(self.phis) - nP * np.sin(self.phis)).flatten()
        nY = (nR * np.sin(self.phis) + nP * np.cos(self.phis)).flatten()
        nZ = nZ.flatten()
        
        B_n = B[...,0]*nX + B[...,1]*nY + B[...,2]*nZ
        B_norm = np.sqrt(B[...,0]**2 + B[...,1]**2 + B[...,2]**2)
        
        normalization = 0.5 * np.sum(N.flatten() * B_norm **2)*self.dtheta*self.dphi*self.nfp
        flux =  0.5 * np.sum(N.flatten() * B_n ** 2)*self.dtheta*self.dphi*self.nfp
        return flux/normalization
        
    def d_quadratic_flux(self,params):
        """
        Computes derivative of normalized quadratic flux integral wrt params
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            d_quadratic_flux (1d array (2*mnmax-1)): derivative of quadratic_flux
                wrt params
        """
        nax = np.newaxis
        B = self.B_from_points(params)
        d_B = self.d_B_from_points(params)
        
        N = self.norm_normal(params).flatten()
        d_N = self.d_norm_normal(params)
        nR, nP, nZ = self.normal(params)
        d_nR, d_nP, d_nZ = self.d_normal(params)
        
        nX = (nR * np.cos(self.phis) - nP * np.sin(self.phis)).flatten()
        nY = (nR * np.sin(self.phis) + nP * np.cos(self.phis)).flatten()
        nZ = nZ.flatten()

        d_nX = np.reshape(d_nR * np.cos(self.phis)[nax,...] - d_nP * np.sin(self.phis)[nax,...] ,
                          (len(params),len(N)))
        d_nY = np.reshape(d_nR * np.sin(self.phis)[nax,...] + d_nP * np.cos(self.phis)[nax,...],
                          (len(params),len(N)))
        d_nZ = np.reshape(d_nZ,(len(params),len(N)))
        d_N = np.reshape(d_N,(len(params),len(N)))
        
        B_n = (B[...,0]*nX + B[...,1]*nY + B[...,2]*nZ)[nax,...]
        B_norm = np.sqrt(B[...,0]**2 + B[...,1]**2 + B[...,2]**2)[nax,...]

        d_B_n = d_B[...,0]*nX[nax,...] + d_B[...,1]*nY[nax,...] + d_B[...,2]*nZ[nax,...] \
            + B[nax,...,0]*d_nX + B[nax,...,1]*d_nY + B[nax,...,2]*d_nZ
        d_B_norm = (B[nax,...,0]*d_B[...,0] + B[nax,...,1]*d_B[...,1] \
                 + B[nax,...,2]*d_B[...,2])/B_norm
        
        normalization = 0.5 * np.sum(N.flatten() * B_norm**2)*self.dtheta*self.dphi*self.nfp
        flux =  0.5 * np.sum(N.flatten() * B_n ** 2)*self.dtheta*self.dphi*self.nfp

        d_flux = 0.5 * np.sum(d_N * B_n** 2 
                         + 2 * N[nax,...] * B_n * d_B_n,axis=1)*self.dtheta*self.dphi*self.nfp
        d_normalization = 0.5 * np.sum(d_N * B_norm ** 2 
                         + 2 * N[nax,...] * B_norm * d_B_norm,axis=1)*self.dtheta*self.dphi*self.nfp

        return d_flux/normalization - flux*d_normalization/(normalization*normalization)
    
    def A_from_points(self,params):
        """
        Computes vector potential on angular grid on surface computed from 
            provided parameters
            
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs;
            A (2d array (size(thetas), 3)): Cartesian components of vector
                potential on flattened angular grid
        """
        R, Z = self.position(params)
        X = R * np.cos(self.phis)
        Y = R * np.sin(self.phis)
        
        points = np.zeros((len(X.flatten()), 3))
        points[:,0] = X.flatten()
        points[:,1] = Y.flatten()
        points[:,2] = Z.flatten()
        self.biotsavart.set_points(points)
        return self.biotsavart.A        
    
    def toroidal_flux(self,params):
        """
        Computes toroidally averaged toroidal flux through given surface
        
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs;
            flux (float): toroidally averaged toroidal flux      
        """
        A = self.A_from_points(params)
        
        dRdtheta, dRdphi, dZdtheta, dZdphi = self.position_derivatives(params)
        dXdtheta = (dRdtheta * np.cos(self.phis)).flatten()
        dYdtheta = (dRdtheta * np.sin(self.phis)).flatten()
        dZdtheta = dZdtheta.flatten()
        
        A_dot_drdtheta = A[...,0]*dXdtheta + A[...,1]*dYdtheta + A[...,2]*dZdtheta
        
        return np.sum(A_dot_drdtheta)*self.dphi*self.dtheta*self.nfp/(2*np.pi)     
    
    def ft_surface(self,params,mmax,nmax):
        """
        Performs Fourier transform of cylindrical coordinates and saves to a file
            in "VMEC format"
            
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
            mmax (int) : maximum poloidal mode number for FT
            nmax (int) : maximum toroidal mode number for FT
        Outputs:
            Rbc (1d array (mnmax)): Fourier harmonics for radius
            Zbs (1d array (mnmax)): Fourier harmonics for height
        """
        params = self.params_full(params)
        paramsR = params[0:self.mnmax]
        paramsZ = params[self.mnmax::]
        
        mnmax, xm, xn = self.init_modes(mmax,nmax)
        xn = xn*self.nfp
        Rbc = np.zeros((mnmax))
        Zbs = np.zeros((mnmax))
        if (mnmax < self.mnmax):
            for im in range(mnmax):
                Rbc[im] = paramsR[(self.xm==xm[im])*(self.xn==xn[im])]
                Zbs[im] = paramsZ[(self.xm==xm[im])*(self.xn==xn[im])]
        elif (mnmax > self.mnmax):
            for im in range(self.mnmax):
                Rbc[(self.xm[im]==xm)*(self.xn[im]==xn)] = paramsR[im]
                Zbs[(self.xm[im]==xm)*(self.xn[im]==xn)] = paramsZ[im]      
        else:
            Rbc = paramsR
            Zbs = paramsZ
        
        f = open("boundary.txt","w")
        for im in range(mnmax):
            f.write('Rbc(%d,%d) = %0.12f\n' % (xn[im]/self.nfp,xm[im],Rbc[im]))
            f.write('Zbs(%d,%d) = %0.12f\n' % (xn[im]/self.nfp,xm[im],Zbs[im]))
        f.close()
        
        return Rbc, Zbs
    
    def qfm_metric(self,paramsInit=None,gtol=1e-6):
        """
        Computes minimum of quadratic flux function beginning with initial guess
            paramsInit
            
        Inputs:
            paramsInit (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            fopt (double): minimum objective value
            gtol (double): gradient norm tolerance for BFGS solver
        """
        if paramsInit is None:
            paramsInit = self.paramsPrev
        if (np.ndim(paramsInit)!=1):
            raise ValueError('paramsInit has incorrect dimensions')
        if (len(paramsInit)!=2*self.mnmax-1):
            raise ValueError('paramsInit has incorrect length')

        optimizer = GradOptimizer(len(paramsInit))
        optimizer.add_objective(self.quadratic_flux,self.d_quadratic_flux,1)
        xopt, fopt, result = optimizer.optimize(paramsInit,package='scipy',
                                            method='BFGS',options={'gtol':gtol})
        if (result==0):
            self.paramsPrev = xopt
            return fopt
        else:
            raise RuntimeError('QFM solver not successful!')
            
    def d_qfm_metric_d_coil_coeffs(self,params=None):
        """
        Computes derivative of qfm metric with respect to coil coefficients
            
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            res (1d array (3*ncoils*(2*Nt_coils-1))): derivative of qfm metric
                wrt coil coeffs
        """
        if params is None:
            params = self.paramsPrev
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')

        B = self.B_from_points(params)
        # Shape: (ncoils,npoints,nparams,3)
        dB_by_dcoilcoeff = self.d_B_from_points_dcoilcoeff(params)
        
        N = self.norm_normal(params)
        nR, nP, nZ = self.normal(params)
        
        nX = (nR * np.cos(self.phis) - nP * np.sin(self.phis)).flatten()
        nY = (nR * np.sin(self.phis) + nP * np.cos(self.phis)).flatten()
        nZ = nZ.flatten()
        N = N.flatten()
        B_n = B[...,0]*nX + B[...,1]*nY + B[...,2]*nZ
        B_norm = np.sqrt(B[...,0]**2 + B[...,1]**2 + B[...,2]**2)
        normalization = 0.5 * np.sum(N * B_norm **2)
        flux =  0.5 * np.sum(N * B_n ** 2)
        f = flux/normalization
        
        nax = np.newaxis
        res = []
        for dB in dB_by_dcoilcoeff:
            deltaB_n = dB[...,0]*nX[:,nax] \
                + dB[...,1]*nY[:,nax] \
                + dB[...,2]*nZ[:,nax]
            deltaB2 = dB[...,0]*B[:,nax,0] \
                + dB[...,1]*B[:,nax,1] \
                + dB[...,2]*B[:,nax,2]
            res.append(np.sum(N[:,nax] * deltaB_n * B_n[:,nax],axis=0)/normalization \
            - (f/normalization)*np.sum(N[:,nax] * deltaB2,axis=0))
        res = self.stellarator.reduce_coefficient_derivatives([ires for ires in res])
        return res
        
    def d_qfm_metric_d_coil_currents(self,params=None):
        """
        Computes derivative of qfm metric with respect to coil currents
            
        Inputs:
            params (1d array (2*mnmax-1)): surface Fourier parameters 
                excluding R00
        Outputs:
            res (1d array (ncoils)): derivative of qfm metric
                wrt coil currents
        """
        if params is None:
            params = self.paramsPrev
        if (np.ndim(params)!=1):
            raise ValueError('params has incorrect dimensions')
        if (len(params)!=2*self.mnmax-1):
            raise ValueError('params has incorrect length')

        B = self.B_from_points(params)
        # Shape: (ncoils,npoints,3)
        dB_by_dcoilcurrents = self.d_B_from_points_dcoilcurrents(params)
        
        N = self.norm_normal(params)
        nR, nP, nZ = self.normal(params)
        
        nX = (nR * np.cos(self.phis) - nP * np.sin(self.phis)).flatten()
        nY = (nR * np.sin(self.phis) + nP * np.cos(self.phis)).flatten()
        nZ = nZ.flatten()
        N = N.flatten()
        B_n = B[...,0]*nX + B[...,1]*nY + B[...,2]*nZ
        B_norm = np.sqrt(B[...,0]**2 + B[...,1]**2 + B[...,2]**2)
        normalization = 0.5 * np.sum(N * B_norm **2)
        flux =  0.5 * np.sum(N * B_n ** 2)
        f = flux/normalization
        
        nax = np.newaxis
        res = []
        for dB in dB_by_dcoilcurrents:
            deltaB_n = dB[...,0]*nX \
                + dB[...,1]*nY \
                + dB[...,2]*nZ
            deltaB2 = dB[...,0]*B[:,0] \
                + dB[...,1]*B[:,1] \
                + dB[...,2]*B[:,2]
            res.append(np.sum(N * deltaB_n * B_n,axis=0)/normalization \
            - (f/normalization)*np.sum(N * deltaB2,axis=0))
        res = self.stellarator.reduce_current_derivatives(res)
        return res
        