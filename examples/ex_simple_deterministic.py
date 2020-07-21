from pyplasmaopt import *
from problem2_objective import get_objective
from scipy.optimize import minimize
import numpy as np

#python3 example2_simple.py --mode deterministic --ninsamples 0 --noutsamples 0 --ppp 10 --optimizer bfgs 


obj, args = get_objective()
outdir = "John_test/"
solver = "bfgs"

x = obj.x0
obj.update(x)
obj.callback(x)

max_iterations = 100
memory = 200

def func_scipy(x):
	obj.update(x)
	res = obj.res
	dres = obj.dres
	return res, dres
	
res = minimize(func_scipy, x, jac=True, method=solver, tol=1e-20, options={"maxiter": max_iterations, "maxcor": memory}, callback=obj.callback)


print("Hey look this somehow finished")
