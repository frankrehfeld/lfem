#
# import other modules
#

import numpy as np
import wir108_series4_FEM as lfem
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse import csr_matrix

#
# constructing and defining raw data
#
print("[constructing and defining raw data]")
h0 = 0.1
n = 5
[p, t] = lfem.square(1,h0)
def f(x,y):
    return 0

def g(x,y):
    return x+y
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
bn = lfem.boundaryNodes(p,t)
In = lfem.interiorNodes(p,t)
solution = np.zeros((B.shape[0]))
l = np.delete(l,bn,0)
T = np.eye(B.shape[0])
T = np.delete(T,bn,0)
T = csr_matrix(T)
B_new = T.dot(B).dot(T.transpose())
u = spsolve(B_new,l)
for i in range(len(u)):
    solution[In[i]] = u[i]
for i in range(len(bn)):
    solution[bn[i]] = g(p[bn[i]][0],p[bn[i]][0])
#
# plotting the solution
#
print("[Plotting solution u(x1,x2)]")
lfem.plot(p, t, solution)
