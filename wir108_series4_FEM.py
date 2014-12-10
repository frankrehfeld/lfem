#
# import other modules
#

import numpy as np
# from numpy import transpose
# from numpy import dot
from numpy.linalg import det
from numpy import zeros
from scipy.sparse import lil_matrix
import FEM as FEM

#
# adjoint(A)
#
# computes the adjoint of a 2x2 matrix
#
# input:
# A - 2x2-matrix
#
# output:
# adj - adjoint of A
#
def adjoint(A):
    adj = zeros((2,2))
    adj[0][0] = A[1][1]
    adj[0][1] = -A[0][1]
    adj[1][0] = -A[1][0]
    adj[1][1] = A[0][0]
    adj = np.array(adj)
    return adj

#
# elemStiffness(p)
#
# computes the element stiffness matrix related to the bilinear form
#   a_K(u,v) = int_K grad u . grad v dx
# for linear FEM on triangles.
#
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
#
# output:
# AK - element stiffness matrix
#
def elemStiffness(p):
    # version 1
    # FK = np.array([p[1]-p[0], p[2]-p[0]]).transpose();
    FK = np.column_stack((p[1]-p[0],p[2]-p[0]))
    AK = zeros((3,3));
    for i in range(3):
        for j in range(3):
            #
            # to avoid to much .dot-concatenation
            #
            first = dN(j).dot(adjoint(FK))
            second = adjoint(FK).transpose().dot(dN(i).transpose())
            # AK[i][j] = 1/(2*norm(FK))*dN(j).dot(adjoint(FK).dot(adjoint(FK).transpose().dot(dN(i).transpose())))
            AK[i][j] = 1/(2)*first.dot(second)*np.power(det(FK),-1)
    return AK

#
# dN(i)
#
# computes the gradiant of the i-th elemental shape function
#
# input:
# i - determines the elemental shape function
#
# output:
# grad - grad of N_i
#
def dN(i):
    if i == 0:
        grad = np.array([-1, -1]);
    elif i == 1:
        grad = np.array([1, 0]);
    elif i == 2:
        grad = np.array([0, 1]);
    else:
        print('triangle only has 3 element shape functions');
    return grad

#
# elemMass(p)
#
# computes the element mass matrix related to the bilinear form
#   m_K(u,v) = int_K u v dx
# for linear FEM on triangles.
#
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
#
# output:
# MK - element mass matrix
#
def elemMass(p):
        FK = np.array([p[1]-p[0], p[2]-p[0]]).transpose()
        MK = zeros((3,3))
        for i in range(3):
                for j in range(3):
                        if i == j:
                                MK[i][j] = (1/12)*det(FK)
                        else:
                                MK[i][j] = (1/24)*det(FK)
        return MK

#
# elemLoad(p, n, f)
#
# returns the element load vector related to linear form
#   l_K(v) = int_K f v dx
# for linear FEM on triangles.
#
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function
#
# output:
# fK - element load vector (3x1 array)
#
def elemLoad(p, n, f):
    fK = zeros((3,1))
    # p=np.array(p)
    # Phi = np.zeros((2,2))
    # Phi[:,0]=p[1,:]-p[0,:]
    # Phi[:,1]=p[2,:]-p[0,:]
    # Phi = np.array([p[1]-p[0], p[2]-p[0]])
    Phi = np.column_stack((p[1]-p[0],p[2]-p[0]))
    [x, w] = FEM.gaussTriangle(n)
    z = []
    for i in range(len(x)):
        shifted = list(map(lambda x: (x+1)/2,x[i]))
        z.append(shifted)
    for i in range(3):
        for j in range(len(x)):
            ab = [0,0]
            ab[0] = Phi[0].dot(z[j])+p[0][0]
            ab[1] = Phi[1].dot(z[j])+p[0][1]
            fK[i] = fK[i] + det(Phi)*w[j]*f(ab[0],ab[1])*N(z[j],i)/4
    return fK

#
# N(x,i)
#
# computes the function value of N_i(x)
#
# input:
# x - point on which N_i is evaluated
# i - order of the element shape function which is evaluated
#
# output:
# N_i(x)
#
def N(x,i):
    if i == 0:
        return 1-x[0]-x[1]
    elif i == 1:
        return x[0]
    elif i == 2:
        return x[1]
    else:
        print('triangle only has 3 element shape functions');

#
# stiffness(p, t)
#
# returns the stiffness matrix related to the bilinear form
#   int_Omega grad u . grad v dx
# for linear FEM on triangles.
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
#
# output:
# Stiff - NxN stiffness matrix in scipy's sparse lil format
#
def stiffness(p, t):
    N = len(p)
    Stiff = lil_matrix((N,N))
    for tri in t:
        points = [p[tri[0]], p[tri[1]], p[tri[2]]]
        TK = lil_matrix((3,N))
        for i in range(3):
            TK[i,tri[i]] = 1
        elemStiff = lil_matrix((3,3))
        elemStiff[:3, [0,1,2]] = elemStiffness(points)
        toAdd = TK.transpose().dot(elemStiff).dot(TK)
        Stiff = (Stiff + toAdd)
    return Stiff

#
# mass(p, t)
#
# returns the mass matrix related to the bilinear form
#   int_Omega u v dx
# for linear FEM on triangles.
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
#
# output:
# Mass - NxN mass matrix in scipy's sparse lil format
#
def mass(p, t):
    N = len(p)
    Mass = lil_matrix((N,N))
    for tri in t:
        points = [p[tri[0]], p[tri[1]], p[tri[2]]]
        TK = lil_matrix((3,N))
        for i in range(3):
            TK[i,tri[i]] = 1
        eleMass = lil_matrix((3,3))
        eleMass[:3, [0,1,2]] = elemMass(points)
        toAdd = TK.transpose().dot(eleMass).dot(TK)
        Mass = (Mass + toAdd)
    return Mass

#
# load(p, t, n, f)
#
# returns the load vector related to the linear form
#   int_Omega f v dx
# for linear FEM on triangles.
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function
#
# output:
# Load - Nx1 load
#
def load(p, t, n, f):
    N = len(p)
    Load = zeros((N,1))
    for tri in t:
        points = [p[tri[0]], p[tri[1]], p[tri[2]]]
        TK = lil_matrix((N,3))
        for i in range(3):
            TK[tri[i],i] = 1
        LK = elemLoad(points, n, f)
        Load = Load + TK.dot(LK)
    return Load

#
# boundaryEdges(p, t)
#
# returns the endpoints of boundary edges as indices into p.
#
# input:
# p - Nx2 array with coordinates of the nodes
# t - Mx3 array with indices of nodes of the triangles
#
# output:
# be - Bx2 array of nodes as indices into p that are endpoints of boundary
#      edges.
#
def boundaryEdges(p,t):
    N = len(p)
    adjacent = np.zeros((N,N))
    for tri in t:
        adjacent[tri[0]][tri[1]] += 1
        adjacent[tri[1]][tri[0]] += 1
        adjacent[tri[1]][tri[2]] += 1
        adjacent[tri[2]][tri[1]] += 1
        adjacent[tri[2]][tri[0]] += 1
        adjacent[tri[0]][tri[2]] += 1
    adjacent = np.triu(adjacent,1)
    itemindex = np.where(adjacent==1)
    be = np.zeros((len(itemindex[0]),2))
    for i in range(len(itemindex[0])):
        be[i][0] = itemindex[0][i]
        be[i][1] = itemindex[1][i]
    return be

#
# boundaryNodes(p, t)
#
# returns the nodes positioned on the boundary as indices into p.
#
# input:
# p - Nx2 array with coordinates of the nodes
# t - Mx3 array with indices of nodes of the triangles
#
# output:
# bn - Bx1 array of nodes as indices into p that are positioned on the boundary.
#
def boundaryNodes(p,t):
    be = boundaryEdges(p,t)
    bn = np.unique(be)
    return bn

#
# interiorNodes(p, t)
#
# returns the interior nodes as indices into p.
#
# input:
# p - Nx2 array with coordinates of the nodes
# t - Mx3 array with indices of nodes of the triangles
#
# output:
# In - Ix1 array of nodes as indices into p that do not lie on the
#      boundary
#
def interiorNodes(p, t):
    be = boundaryEdges(p,t)
    bn = np.unique(be)
    nodes = np.linspace(0,len(p)-1,num=len(p))
    In = []
    for node in nodes:
        if not(node in bn):
            In.append(node)
    return In
