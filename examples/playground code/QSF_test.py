from pyplasmaopt import *
from problem2_objective import get_objective
import scipy

obj = get_objective()

qsf = QuasiSymmetricField(2.25, obj.ma)

print(obj.ma.gamma.shape)

