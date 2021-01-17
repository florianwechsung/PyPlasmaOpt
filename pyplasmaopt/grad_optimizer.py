import numbers
import numpy as np
import sys
import scipy.linalg
import scipy.optimize
import nlopt
from mpi4py import MPI

"""
A class used for gradient-based optimization
"""

class GradOptimizer:
    def __init__(self,nparameters,name='optimization'):
        if (isinstance(nparameters,int) == False):
            raise TypeError('nparameters must be a int')
        if (nparameters <= 0):
            raise ValueError('nparameters must be > 0')
        if (isinstance(name,str) == False):
            raise TypeError('name must be an str')  
            
        self.nparameters = nparameters
        self.name = name
        
        self.objectives = []
        self.objective_weights = []
        self.objectives_grad = []
        self.nobjectives = 0
        
        self.bound_constraints_min = []
        self.bound_constraints_max = []
        self.bound_constrained = False
        
        self.n_ineq_constraints = 0
        self.ineq_constraints = []
        self.ineq_constraints_grad = []
        self.ineq_constrained = False

        self.n_eq_constraints = 0
        self.eq_constraints = []
        self.eq_constraints_grad = []
        self.eq_constrained = False
        
        self.parameters_hist = []
        self.objectives_hist = np.zeros([])
        self.objective_hist = []
        self.objectives_grad_norm_hist = []
        self.neval_objectives = 0
        self.neval_objectives_grad = 0 
        
        self.ineq_constraints_hist = np.zeros([])
        self.ineq_constraints_grad_norm_hist = np.zeros([])
        self.neval_ineq_constraints = 0
        self.neval_ineq_constraints_grad = 0
        self.eq_constraints_hist = []
        self.eq_constraints_grad_norm_hist = []
        self.neval_eq_constraints = 0
        self.neval_eq_constraints_grad = 0                

        self.nlopt_methods = ('MMA','SLSQP','CCSAQ','LBFGS','TNEWTON',\
                              'TNEWTON_PRECOND_RESTART','TNEWTON_PRECOND',\
                              'TNEWTON_RESTART','VAR2','VAR1','StOGO',\
                              'STOGO_RAND','MLSL','MLSL_LDS')
        self.nlopt_methods_ineq_constrained = ('MMA','SLSQP','CCSAQ')
        self.nlopt_methods_eq_constrained = ('SLSQP','CCSAQ')
        self.nlopt_methods_bound_constrained = ('SLSQP')
        self.nlopt_dict = {'MMA': nlopt.LD_MMA, 'SLSQP': nlopt.LD_SLSQP, \
          'CCSAQ': nlopt.LD_CCSAQ, 'LBFGS': nlopt.LD_LBFGS, 'TNEWTON': \
          nlopt.LD_TNEWTON, 'TNEWTON_PRECOND_RESTART': \
          nlopt.LD_TNEWTON_PRECOND_RESTART, 'TNEWTON_PRECOND': \
          nlopt.LD_TNEWTON_PRECOND, 'TNEWTON_RESTART': nlopt.LD_TNEWTON_RESTART,\
          'VAR2': nlopt.LD_VAR2, 'VAR1': nlopt.LD_VAR1, 'SToGO': nlopt.GD_STOGO, \
          'STOGO_RAND': nlopt.GD_STOGO_RAND}
    
        self.scipy_methods = ('CG','BFGS','Newton-CG','L-BFGS-B','TNC','SLSQP',\
                              'dogleg','trust-ncg','trust-krylov','trust-exact',\
                              'trust-constr')
        self.scipy_methods_eq_constrained = ('SLSQP','trust-constr')
        self.scipy_methods_ineq_constrained = ('SLSQP','trust-constr')
        self.scipy_methods_bound_constrained = ('L-BFGS-B','TNC','SLSQP')
        
    def add_objective(self,objective,objective_grad,weight=1):
        """
        Adds objective function to optimization problem, which will be scalarized
            according to the specified weights. 
            
            f(x) = \sum_i w_i f_i(x)
        
        Args:
            objective (function): objective function to add (f_i(x))
            objective_grad (function): gradient of objective function  (f_i'(x))
            weight (scalar): weight for objective function in scalarized 
                objective function (w_i)
        """
        self._test_function(objective,'objective')
        self._test_function(objective_grad,'objective_grad')
        self._test_scalar(weight,'weight')
            
        self.nobjectives += 1
        self.objectives.append(objective)
        self.objectives_grad.append(objective_grad)
        self.objective_weights.append(weight)
        
    def remove_objectives(self):
        """
        Remove all objective functions from optimization problem
        """
        self.objectives = []
        self.objective_weights = []
        self.objectives_grad = []
        self.nobjectives = 0
        
    def add_bound(self,bound,min_or_max='min'):
        """
        Adds bound constraint to optimization problem

            bound_min <= x <= bound_max
        
        Args:
            bound (scalar or list): bound constraint 
            min_or_max (str): should be 'max' or 'min'. Indicates
                maximum or minimum bound constraint.
        """
        if (isinstance(bound, (numbers.Number,list)) == False):
            raise TypeError('bound must be a scalar or list')
        if (isinstance(bound, list)):
            if (len(bound) != self.nparameters):
                raise ValueError('bound must have same length as Nparameters')
        if (isinstance(min_or_max, str) == False):
            raise TypeError('min_or_max must be a str')
        if (min_or_max not in ('min', 'max')):
            raise ValueError("min_or_max must be 'min' or 'max'")
        
        self.bound_constrained = True
        if (min_or_max == 'min'):
            if (isinstance(bound,list)):
                self.bound_constraints_min = bound
            else:
                self.bound_constraints_min = \
                    [bound for i in range(self.nparameters)]
        if (min_or_max == 'max'):
            if (isinstance(bound,list)):
                self.bound_constraints_max = bound
            else:
                self.bound_constraints_max = \
                    [bound for i in range(self.nparameters)]
                
    def remove_bounds(self):
        """
        Remove all bound constraints from optimization problem
        """
        self.bound_constraints_min = []
        self.bound_constraints_max = []
        self.bound_constrained = False
            
    def add_ineq(self,ineq_constraint,ineq_constraint_grad):
        """
        Adds inequality constraint to optimization problem
        
            g_i(x) >= 0
        
        Args:
            ineq_constraint (function): function defining inequality constraint 
                (g_i(x))
            ineq_constraint_grad (function): function defining ineqauality
                constraint gradient (g_i'(x))
        """
        self._test_function(ineq_constraint,'ineq_constraint')
        self._test_function(ineq_constraint_grad,'ineq_constraint_grad')
            
        self.ineq_constrained = True
        self.n_ineq_constraints += 1
        self.ineq_constraints.append(ineq_constraint)
        self.ineq_constraints_grad.append(ineq_constraint_grad)
        
    def remove_ineq(self):
        """
        Remove all inequality constraints from optimization problem
        """
        
        self.n_ineq_constraints = 0
        self.ineq_constraints = []
        self.ineq_constraints_grad = []
        self.ineq_constrained = False

    def add_eq(self,eq_constraint,eq_constraint_grad):
        """
        Adds equality constraint to optimization problem
        
            h_i(x) = 0
        
        Args:
            eq_constraint (function): function defining equality constraint 
                (h_i(x))
            eq_constraint_grad (function): function defining eqauality
                constraint gradient (h_i'(x))
        """
        self._test_function(eq_constraint,'eq_constraint')
        self._test_function(eq_constraint_grad,'eq_constraint_grad')
            
        self.eq_constrained = True
        self.n_eq_constraints += 1
        self.eq_constraints.append(eq_constraint)
        self.eq_constraints_grad.append(eq_constraint_grad)
        
    def remove_eq(self):
        """
        Remove all equality constraints from optimization problem
        """
        self.n_eq_constraints = 0
        self.eq_constraints = []
        self.eq_constraints_grad = []
        self.eq_constrained = False
        
    def reset_history(self):
        """
        Reset all function evaluation history
        """  
        self.objectives_hist = np.zeros([])
        self.objective_hist = []
        self.objectives_grad_norm_hist = []
        self.neval_objectives = 0
        self.neval_objectives_grad = 0 
        self.ineq_constraints_hist = np.zeros([])
        self.ineq_constraints_grad_norm_hist = np.zeros([])
        self.neval_ineq_constraints = 0
        self.neval_ineq_constraints_grad = 0
        self.eq_constraints_hist = []
        self.eq_constraints_grad_norm_hist = []
        self.neval_eq_constraints = 0
        self.neval_eq_constraints_grad = 0
        
    def reset_all(self):
        """
        Reset all function evaluation history, constraints, bounds, and 
        objectives.
        """  
        self.reset_history()
        self.remove_eq()
        self.remove_ineq()
        self.remove_bounds()
        self.remove_objectives()

    def objectives_fun(self,x):
        """
        Calls objective functions and scalarizes according to specified
            weights
            
            f(x) = \sum_i w_i f_i(x)
        
        Args:
            x (list/array): parameters at which to evaluate objective function.
                Should have length = nparameters
        Returns: 
            objective (float): value of scalarized objective
        """
        self._test_x(x)
        # Check that no new objectives have been added since previous calls
        if (self.objectives_hist.ndim == 2):
            if (len(self.objectives_hist[0,:]) != self.nobjectives):
                raise RuntimeError('''Number of objectives has been increased 
                                   since previous call to objectives_fun''')
        objective_values = np.zeros(self.nobjectives)
        for i in range(self.nobjectives):
            objective_values[i] = self.objectives[i](x)
        if (self.neval_objectives == 0):
            self.objectives_hist = np.zeros((1,self.nobjectives))
            self.objectives_hist[0,:] = objective_values
            self.parameters_hist = np.zeros((1,self.nparameters))
            self.parameters_hist[0,:] = x
        else:
            self.objectives_hist = np.vstack((np.array(self.objectives_hist),\
                                                objective_values))
            # Make sure x is a row vector 
            xarr = np.zeros([1,len(x)])
            xarr[0,:] = x
            self.parameters_hist = np.vstack((np.array(self.parameters_hist),\
                                              xarr))
        objective = np.dot(np.array(self.objective_weights),\
                           np.array(objective_values))
        self.objective_hist.append(objective)
        self.neval_objectives += 1
        
        rank = MPI.COMM_WORLD.Get_rank()
        if (rank==0):
            np.savetxt('objectives_hist.txt',self.objectives_hist)
            np.savetxt('objective_hist.txt',self.objective_hist)
            np.savetxt('parameters_hist.txt',self.parameters_hist)
        
        return objective
                
    def objectives_grad_fun(self,x):
        """
        Calls objective gradient functions and multiply with weights for
            computing gradient of scalarized objective function
            
            f'(x) = \sum_i w_i f_i'(x)
            
        Args:
            x (list/array):  parameters at which to evaluate objective function
                gradient. Should have length = nparameters
                
        Returns: 
            objective_grad (list/array): gradient of objective function. 
                Has length = nparameters.
        """
        self._test_x(x)
        # Check that no new objectives have been added since previous calls
        if (self.objectives_hist.ndim == 2):
            if (len(self.objectives_hist[0,:]) != self.nobjectives):
                raise RuntimeError('''Number of objectives has been increased 
                                   since previous call to objectives_fun''')
        
        objectives_grad_value = np.zeros((self.nparameters,self.nobjectives))
        for i in range(self.nobjectives):
            objectives_grad_value[:,i] = self.objectives_grad[i](x)
        objective_grad = np.matmul(np.array(objectives_grad_value),\
                                   np.array(self.objective_weights))
        grad_norm = scipy.linalg.norm(objective_grad)
        self.objectives_grad_norm_hist.append(grad_norm)
        self.neval_objectives_grad += 1
        
        rank = MPI.COMM_WORLD.Get_rank()
        if (rank==0):
            np.savetxt('objectives_grad_norm_hist.txt',self.objectives_grad_norm_hist)
        
        return objective_grad
    
    def ineq_fun(self,x):
        """
        Calls inequality constraint function 
            g_i(x) >= 0
        
        Args:
            x (list/array): parameters at which to evaluate inequality function.
                Should have length = nparameters
        Returns: 
            ineq_value (list): value of inequality constraint functions
        """
        self._test_x(x)
        # Check that no new constraints have been added since previous calls
        if (self.ineq_constraints_hist.ndim == 2):
            if (len(self.ineq_constraints_hist[0,:]) != self.n_ineq_constraints):
                raise RuntimeError('''Number of constraints has been increased 
                                   since previous call to ineq_fun''')
        ineq_values = np.zeros(self.n_ineq_constraints)
        for i in range(self.n_ineq_constraints):
            ineq_values[i] = self.ineq_constraints[i](x)
        if (self.neval_ineq_constraints == 0):
            self.ineq_constraints_hist = np.zeros((1,self.n_ineq_constraints))
            self.ineq_constraints_hist[0,:] = ineq_values
        else:
            self.ineq_constraints_hist = \
                            np.vstack((np.array(self.ineq_constraints_hist),\
                                                ineq_values))
        self.neval_ineq_constraints += 1
        return ineq_values
    
    def ineq_grad_fun(self,x):
        """
        Computes gradient of inequality constraint function 
            g'(x)
        
        Args:
            x (list/array): parameters at which to evaluate inequality gradient.
                Should have length nparameters.
        Returns: 
            ineq_gradient (array): value of inequality gradient function.
                Has shape (nparameters,n_ineq_constraints)
        """
        self._test_x(x)
        # Check that no new constraints have been added since previous calls
        if (self.ineq_constraints_grad_norm_hist.ndim == 2):
            if (len(self.ineq_constraints_grad_norm_hist[0,:]) != \
                self.n_ineq_constraints):
                raise RuntimeError('''Number of constraints has been increased 
                                   since previous call to ineq_grad_fun''')
        ineq_grad = np.zeros([self.nparameters,self.n_ineq_constraints])
        for i in range(self.n_ineq_constraints):
            ineq_grad[:,i] = self.ineq_constraints_grad[i](x)
        if (self.neval_ineq_constraints_grad == 0):
            self.ineq_constraints_grad_norm_hist = np.zeros((1,self.n_ineq_constraints))
            self.ineq_constraints_grad_norm_hist[0,:] = scipy.linalg.norm(ineq_grad)
        else:
            self.ineq_constraints_grad_norm_hist = \
                     np.vstack((np.array(self.ineq_constraints_grad_norm_hist),\
                                                ineq_grad))
        self.neval_ineq_constraints_grad += 1
        return ineq_grad.T
    
    def optimize(self,x,package='nlopt',method='CCSAQ',ftol_abs=1e-8,
                 ftol_rel=1e-8,xtol_abs=1e-8,xtol_rel=1e-8,tol=1e-8,**kwargs):
        """
        Optimizes scalarized objective function using nlopt or scipy package
            
        Args:
            x (list/array): initial parameters at which to evaluate objective 
                function. Should have length = nparameters
            package (str): should be 'nlopt' or 'scipy'. Package from which 
                optimization method will be chosen
            method (str): optimization algorithm to use
            ftol_abs (float): absolute tolerance in function value
            ftol_rel (float): relative tolerance in function value
            xtol_abs (float): absolute tolerance in parameters
            xtol_rel (float): relative tolerance in parameters
            tol (float): tolerance for scipy
        
        Returns: 
            xopt (list/array): final parameters evaluated during optimization
            fopt (float): final ojective function value
            result (int): return value from scipy/nlopt providing
                reason for termination
        """
        self._test_x(x)
        self._test_scalar(ftol_abs,'ftol_abs')
        self._test_scalar(ftol_rel,'ftol_rel')
        self._test_scalar(xtol_abs,'xtol_abs')
        self._test_scalar(xtol_rel,'xtol_rel')

        if (package not in ('nlopt','scipy')):
            raise ValueError("package must be ['nlopt','scipy']")
            
        if (package == 'nlopt'):
            self._test_method_nlopt(method)
            [xopt, fopt, result] = self.nlopt_optimize(x,method,ftol_abs,ftol_rel,\
                                       xtol_abs,xtol_rel,**kwargs)
        if (package == 'scipy'):
            self._test_method_scipy(method)
            [xopt, fopt, result] = self.scipy_optimize(x,method,**kwargs)
            
        # Save output 
        rank = MPI.COMM_WORLD.Get_rank()
        if (rank==0):
            np.savetxt('xopt.txt',xopt)
            np.savetxt('fopt.txt',[fopt])
            np.savetxt('result.txt',[result])
            np.savetxt('parameters_hist.txt',self.parameters_hist)
            np.savetxt('objectives_hist.txt',self.objectives_hist)
            np.savetxt('objective_hist.txt',self.objective_hist)
            np.savetxt('objectives_grad_norm_hist.txt',self.objectives_grad_norm_hist)
            
        return xopt, fopt, result
    
    def nlopt_objective(self, x, grad):
        """
        Scalarized objective function in format required by nlopt
        
        Args:
            x (list/array): parameters for evaluation. Should have length 
                nparameters.
            grad (list/array): gradient of function at x. Should be set in place
                if not empty.
        Returns:
            objective_value (float): value of objective function 
        """
        objective_value = self.objectives_fun(x)
        if grad.size > 0:
            if (objective_value != 1e12):
                grad[:] = self.objectives_grad_fun(x)
            else:
                grad[:] = 1e12*np.ones(self.nparameters)
        return objective_value
    
    def nlopt_ineq_m(self, result, x, grad):
        """
        Inequality constraint function in format required by nlopt when vector 
        of inequality constraints is imposed. 
        Note that sign has been flipped -> nlopt convention is g <= 0
        
        Args:
            result (list/array): vector inequality function evaluation 
            x (list/array): parameters for evaluation. Should have length 
                nparameters.
            grad (list/array): gradient of function at x. First dimension of array
                is n_ineq_constraints. Second dimension is nparameters. Should be 
                set in place if not empty.
        """
        result[:] = -self.ineq_fun(x)
        if grad.size > 0:
            if (np.all(result!=1e12)):
                grad[:,:] = -self.ineq_grad_fun(x)
            else:
                grad[:,:] = np.zeros([self.n_ineq_constraints,self.nparameters])
            
    def nlopt_ineq(self, x, grad):
        """
        Inequality constraint function in format required by nlopt when single
        inequality constraint is imposed. 
        Note that sign has been flipped -> nlopt convention is g <= 0
        
        Args:
            x (list/array): parameters for evaluation. Should have length 
                nparameters.
            grad (list/array): gradient of function at x. Should be set in place
                if not empty.
        Returns:
            ineq_value (float): value of g_i(x) 
        """
        if (self.n_ineq_constraints > 1):
            raise RuntimeErrr('''nlopt_ineq should only be called if 
                n_ineq_constraints = 1''')
        ineq_value = -self.ineq_fun(x)[0]
        if grad.size > 0:
            if (ineq_value != 1e12):
                grad[:] = -self.ineq_grad_fun(x)[0,:]
            else:
                grad[:] = np.zeros(self.nparameters)
        return ineq_value
    
    def nlopt_eq(self, x, grad):
        """
        Equality constraint function in format required by nlopt for single
        constraint. 
        
        Args:
            x (list/array): parameters for evaluation. Should have length 
                nparameters.
            grad (list/array): gradient of function at x. Should be set in place
                if not empty.
        Returns:
            eq_value (float): value of h_i(x) 
        """
        if (self.n_eq_constraints > 1):
            raise RuntimeErrr('''nlopt_eq should only be called if 
                n_eq_constraints = 1''')
        eq_value = self.eq_constraints[0](x)
        if grad.size > 0:
            if (eq_value != 1e12):
                grad[:] = self.eq_constraints_grad[0](x)
            else:
                grad[:] = np.zeros(self.nparameters)
        return eq_value
    
    def nlopt_eq_m(self, result, x, grad):
        """
        Equality constraint function in format required by nlopt when vector 
        of constraints is imposed. 
        
        Args:
            result (list/array): vector of equality function evaluation [h_i(x)]
            x (list/array): parameters for evaluation. Should have length 
                nparameters.
            grad (list/array): gradient of equality function at x. First 
                dimension of array is n_eq_constraints. Second dimension is 
                nparameters. Should be set in place if not empty.
        """
        for i in range(self.n_eq_constraints):
            result[i] = self.eq_constraints[i](x)
            if grad.size > 0:
                if (np.all(result != 1e12)):
                    grad[i,:] = np.array(self.eq_constraints_grad[i](x))
                else:
                    grad[:,:] = np.zeros([self.n_eq_constraints,self.nparameters])
    
    def nlopt_optimize(self,x,method='SLSQP',ftol_abs=1e-8,ftol_rel=1e-8,\
                       xtol_abs=1e-8,xtol_rel=1e-8,ineq_tol=1e-8,eq_tol=1e-8):
        """
        Optimize objective function with nlopt
        
        Args:
            x (list/array): parameters for evaluation. Should have length 
                nparameters.
            method (str): optimization algorithm to use. Must be in 
                nlopt_methods.
            ftol_abs (float): absolute tolerance in function value
            ftol_rel (float): relative tolerance in function value
            xtol_abs (float): absolute tolerance in parameters
            xtol_rel (float): relative tolerance in parameters
            ineq_tol (float): tolerance in inequality constraint
            eq_tol (float): tolerance in equality constraint
        Returns: 
            xopt (list/array): final parameters evaluated during optimization
            fopt (float): final ojective function value
            result (int): return value from scipy/nlopt providing
                reason for termination

        """
        self._test_method_nlopt(method)
        self._test_scalar(ftol_abs,'ftol_abs')
        self._test_scalar(ftol_rel,'ftol_rel')
        self._test_scalar(xtol_abs,'xtol_abs')
        self._test_scalar(xtol_rel,'xtol_rel')
        self._test_scalar(ineq_tol,'ineq_tol')

        # Use auglag if necessary
        if (self.ineq_constrained and \
                method not in self.nlopt_methods_ineq_constrained):
            auglag = True
            algorithm = nlopt.LD_AUGLAG
            local_opt = nlopt.opt(self.nlopt_dict[method], self.nparameters)
            local_opt.set_ftol_rel(ftol_rel)
            local_opt.set_ftol_abs(ftol_abs)
            local_opt.set_xtol_rel(xtol_rel)
            local_opt.set_xtol_abs(xtol_abs)
        elif (self.eq_constrained and method not in \
                self.nlopt_methods_eq_constrained):
            auglag = True
            algorithm = nlopt.LD_AUGLAG_EQ
            local_opt = nlopt.opt(self.nlopt_dict[method], self.nparameters)
            local_opt.set_ftol_rel(ftol_rel)
            local_opt.set_ftol_abs(ftol_abs)
            local_opt.set_xtol_rel(xtol_rel)
            local_opt.set_xtol_abs(xtol_abs)
        else:
            auglag = False
            algorithm = self.nlopt_dict[method]
            
        opt = nlopt.opt(algorithm, self.nparameters)
        if (self.ineq_constrained):
            if (self.n_ineq_constraints > 1):
                opt.add_inequality_mconstraint(self.nlopt_ineq_m, \
                                      ineq_tol*np.ones(self.n_ineq_constraints))
            else:
                opt.add_inequality_constraint(self.nlopt_ineq, ineq_tol)
        if (self.eq_constrained):
            if (self.n_eq_constraints > 1):
                opt.add_equality_mconstraint(self.nlopt_eq_m, \
                                    eq_tol*np.ones(self.n_eq_constraints))
            else:
                opt.add_equality_constraint(self.nlopt_eq, eq_tol)
        if (auglag):
            opt.set_local_optimizer(local_opt)

        # Set tolerance parameters
        opt.set_ftol_rel(ftol_rel)
        opt.set_ftol_abs(ftol_abs)
        opt.set_xtol_rel(xtol_rel)
        opt.set_xtol_abs(xtol_abs)
        opt.set_min_objective(self.nlopt_objective)
        if (len(self.bound_constraints_min)>0):
            opt.set_lower_bounds(self.bound_constraints_min)
        if (len(self.bound_constraints_max)>0):
            opt.set_upper_bounds(self.bound_constraints_max)
        try: 
            xopt = opt.optimize(x)
        except:
            print('Nlopt completed with an error')
        xopt = self.parameters_hist[-1,:]
        fopt = opt.last_optimum_value()
        result = opt.last_optimize_result()
        return xopt, fopt, result
    
    def scipy_optimize(self,x,method='BFGS',**kwargs):
        """
        Optimize objective function with scipy
        
        Args:
            x (list/array): parameters for evaluation. Should have length 
                nparameters.
            method (str): optimization algorithm to use. Must be in 
                nlopt_methods.
            **kwargs : additional keyword args to be passed to scipy.optimize.minimize
        Returns: 
            xopt (list/array): final parameters evaluated during optimization
            fopt (float): final ojective function value
            result (int): return value from scipy/nlopt providing
                reason for termination
        """

        self._test_method_scipy(method)

        if (self.bound_constrained):
            if (len(self.bound_constraints_min)>0):
                bound_constraints_min = self.bound_constraints_min
            else:
                bound_constraints_min = -np.inf*np.ones(np.shape(x))
            if (len(self.bound_constraints_max)>0):
                bound_constraints_max = self.bound_constraints_max
            else:
                bound_constraints_max = np.inf*np.ones(np.shape(x))
            bounds = scipy.optimize.Bounds(bound_constraints_min,\
                                           bound_constraints_max)
        else:
            bounds = None
        if (self.ineq_constrained):
            ineq_constraints = []
            if (method == 'trust-constr'):
                ineq_constraints.append(scipy.optimize.NonlinearConstraint(\
                               self.ineq_fun,0,np.infty,jac=self.ineq_grad_fun))
            else:
                ineq_constraints.append({'type' : 'ineq','fun':self.ineq_fun,\
                                        'jac': self.ineq_grad_fun})

        if (self.eq_constrained):
            eq_constraints = []
            if (method == 'trust-constr'):
                for i in range(self.n_eq_constraints):
                    eq_constraints.append(scipy.optimize.NonlinearConstraint(\
                                          self.eq_constraints[i],0,0,\
                                          jac = self.eq_constraints_grad[i]))
            else:
                for i in range(self.n_eq_constraints):
                    eq_constraints.append({'type' : 'eq', \
                                           'fun': self.eq_constraints[i], \
                                           'jac': self.eq_constraints_grad[i]})
        if (self.ineq_constrained and self.eq_constrained):
            constraints = eq_constraints.append(ineq_constraints)
        elif (self.ineq_constrained):
            constraints = ineq_constraints
        elif (self.eq_constrained):
            constraints = eq_constraints
        else:
            constraints = None

        OptimizeResult = scipy.optimize.minimize(self.objectives_fun, x, \
                   method=method,jac=self.objectives_grad_fun,bounds=bounds,\
                   constraints = constraints, **kwargs) 
        rank = MPI.COMM_WORLD.Get_rank()
        if (rank==0):
            print(OptimizeResult.message)
        xopt = OptimizeResult.x
        result = OptimizeResult.status
        fopt = OptimizeResult.fun
        return xopt, fopt, result
        
    def _test_x(self,x):
        """
        Test that x has correct dimensions
        """
        if (len(x) != self.nparameters):
            raise ValueError('Incorrect dimension of x')
            
    def _test_scalar(self,scalar,scalar_name):
        """
        Test that scalar_name is a scalar
        """
        if (not isinstance(scalar,numbers.Number)):
            raise TypeError(scalar_name+' must be a scalar')
            
    def _test_function(self,function,function_name):
        """
        Test that function_name is a function
        """
        if (not callable(function)):
            raise TypeError(function_name+' must be a function')
                        
    def _test_method_nlopt(self,method):
        """
        Test that nlopt method matches with specified constraints
        """
        if (method not in self.nlopt_methods):
            raise ValueError('method must be in '+str(self.nlopt_methods))
        if (self.ineq_constrained and self.eq_constrained):
            if (method not in self.nlopt_methods_ineq_constrained):
                raise ValueError('method must be in '+\
                                 str(self.nlopt_methods_ineq_constrained))
            if (method not in self.nlopt_methods_eq_constrained):
                raise ValueError('method must be in '+\
                                 str(self.nlopt_methods_eq_constrained))
        if (self.bound_constrained):
            if (method not in self.nlopt_methods_bound_constrained):
                raise ValueError('method must be in '+\
                                str(self.nlopt_methods_bound_constrained))
                
    def _test_method_scipy(self,method):
        """
        Test that scipy method matches with specified constraints
        """
        if (method not in self.scipy_methods):
            raise ValueError('method must be in '+str(self.scipy_methods))
        if (self.ineq_constrained):
            if (method not in self.scipy_methods_ineq_constrained):
                raise ValueError('method must be in'+\
                                 str(self.scipy_methods_ineq_constrained))
        if (self.eq_constrained):
            if (method not in self.scipy_methods_eq_constrained):
                raise ValueError('method must be in'+\
                                str(self.scipy_methods_eq_constrained))
        if (self.bound_constrained):
            if (method not in self.scipy_methods_bound_constrained):
                raise ValueError('method must be in'+\
                                str(self.scipy_methods_bound_constrrained))


                 

              
  
