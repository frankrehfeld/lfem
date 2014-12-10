#
# import other modules
#

import numpy as np
import wir108_series4_FEM as lfem
from scipy.sparse.linalg import spsolve

#
# constructing and defining raw data
#
print("[constructing and defining raw data]")
h0 = 0.1
n = 3
[p, t] = lfem.square(1,h0)
# print(type(p))
def f(x,y):
    return np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*(1+8*np.power(np.pi,2))
#    return x*y
#
# calculating the system to solve
#
print("[Calculating the system Bu = l]")
cStiff = lfem.stiffness(p,t)
cMass = lfem.mass(p,t)
B = cMass+cStiff
l = lfem.load(p, t, n, f)
#
# Solving the system
#
print("[Solving the system Bu = l]")
u = spsolve(B,l)
#
# plotting the solution
#
print("[Plotting solution u(x1,x2)]")
print(type(u))
lfem.plot(p, t, u)
