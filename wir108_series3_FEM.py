#
# import other modules
#

import numpy as np
# from numpy import transpose
# from numpy import dot
from numpy.linalg import det
from numpy import zeros
from scipy.sparse import lil_matrix

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
    FK = np.array([p[1]-p[0], p[2]-p[0]]).transpose();
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
    #
    # version 2
    # FK = np.array([p[1]-p[0], p[2]-p[0]]).transpose()
    # DK = zeros((2,3))
    # DK[0][0] = p[1][1]-p[2][1]
    # DK[0][1] = p[2][1]-p[0][1]
    # DK[0][2] = p[0][1]-p[1][1]
    # DK[1][0] = p[2][0]-p[1][0]
    # DK[1][1] = p[0][0]-p[2][0]
    # DK[1][2] = p[1][0]-p[0][0]
    # AK = DK.transpose().dot(DK)/(2*det(FK))
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
    import FEM as FEM
    fK = zeros((3,1))
    Phi = np.array([p[1]-p[0], p[2]-p[0]])
    [x, w] = FEM.gaussTriangle(n)
    z = []
    for i in range(len(x)):
        shifted = list(map(lambda x: (x+1)/2,x[i]))
        z.append(shifted)
    for i in range(3):
        for j in range(len(x)):
            ab = [0,0]
            ab[0] = Phi[0].dot(z[j])
            ab[1] = Phi[1].dot(z[j])
            fK[i] = fK[i] + 1/4*w[j]*f(ab[0],ab[1])*N(z[j],i)
    return fK

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
    for triangle in t:
        points = [p[triangle[0]], p[triangle[1]], p[triangle[2]]]
        TK = lil_matrix((3,N))
        for i in range(3):
            TK[i,triangle[i]] = 1
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
    for triangle in t:
        points = [p[triangle[0]], p[triangle[1]], p[triangle[2]]]
        TK = lil_matrix((3,N))
        for i in range(3):
            TK[i,triangle[i]] = 1
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
    for triangle in t:
        points = [p[triangle[0]], p[triangle[1]], p[triangle[2]]]
        TK = lil_matrix((N,3))
        for i in range(3):
            TK[triangle[i],i] = 1
        LK = elemLoad(points, n, f)
        Load = Load + TK.dot(LK)
    return Load

#
# gaussTriangle(n)
#
# returns absiccas and weights for "Gauss integration" in the triangle with
# vertices (-1,-1), (1,-1), (-1,1)
#
# input:
# n - order of the numerical integration (1 <= n <= 5)
#
# output:
# x - 2xp-array of points, where p is the number of points
# w - 1xp-array of weights, where p is the number of points
#
def gaussTriangle(n):

  if n == 1:
      x = [[-1/3., -1/3.]];
      w = [2.];
  elif n == 2:
      x = [[-2/3., -2/3.],
           [-2/3.,  1/3.],
           [ 1/3., -2/3.]];
      w = [2/3.,
           2/3.,
           2/3.];
  elif n == 3:
      x = [[-1/3., -1/3.],
           [-0.6, -0.6],
           [-0.6,  0.2],
           [ 0.2, -0.6]];
      w = [-1.125,
            1.041666666666667,
            1.041666666666667,
            1.041666666666667];
  elif n == 4:
      x = [[-0.108103018168070, -0.108103018168070],
           [-0.108103018168070, -0.783793963663860],
           [-0.783793963663860, -0.108103018168070],
           [-0.816847572980458, -0.816847572980458],
           [-0.816847572980458,  0.633695145960918],
           [ 0.633695145960918, -0.816847572980458]];
      w = [0.446763179356022,
           0.446763179356022,
           0.446763179356022,
           0.219903487310644,
           0.219903487310644,
           0.219903487310644];
  elif n == 5:
      x = [[-0.333333333333333, -0.333333333333333],
           [-0.059715871789770, -0.059715871789770],
           [-0.059715871789770, -0.880568256420460],
           [-0.880568256420460, -0.059715871789770],
           [-0.797426985353088, -0.797426985353088],
           [-0.797426985353088,  0.594853970706174],
           [ 0.594853970706174, -0.797426985353088]];
      w = [0.450000000000000,
           0.264788305577012,
           0.264788305577012,
           0.264788305577012,
           0.251878361089654,
           0.251878361089654,
           0.251878361089654];
  else:
      print('numerical integration of order ' + str(n) + 'not available');

  return x, w


#
# plot(p,t,u)
#
# plots the linear FE function u on the triangulation t with nodes p.
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
# u  - Nx1 coefficient vector
#
def plot(p,t,u):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_trisurf(p[:, 0], p[:, 1], t, u, cmap=plt.cm.jet)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('u')
  plt.show()


# meshes

#
# import other modules
#
import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from math import pow

#
# round_trip_connect(a, b)
#
# Auxiliary function that returns a set of pairs of adjacent indices between
# the integers a and b. This set also includes the pair (b, a) as last entry.
#
# input:
# a - integer
# b - integer (b>a)
#
# output:
# set of pairs of adjacent indices between a and b and the pair (b, a)
#
def round_trip_connect(a, b):
  return [(i, i+1) for i in range(a, b)] + [(b, a)]


#
# max_edge_length(vertices)
#
# Auxiliary function that returns the maximal distance of the three vertices,
# or equivalenty, the maximal length of all edges in the triangle spanned by
# the three vertices.
#
# input:
# vertices - array of three 2d vertices
#
# output:
# maximal distance of the three vertices
#
def max_edge_length(vertices):
  p = np.array(vertices)
  return max(la.norm(p[0]-p[1]),la.norm(p[1]-p[2]),la.norm(p[2]-p[0]))


#
# square(a,h0)
#
# Function that produces an unstructured mesh of a square. The square has side
# length a and its lower left corner is positioned at the origin. The maximal
# mesh width of the produced mesh is smaller or equal to h0. (The maximal mesh
# width of a mesh is the maximal distance of two adjacent vertices, or
# equivalenty, the maximal length of all edges in the mesh.)
#
# input:
# a  - side length of square
# h0 - upper bound for maximal mesh width
#
# output:
# p - Numpy-array of size Nx2 with the x- and y-coordinates of the N vertices
#     in the mesh
# t - Numpy-array of size Mx3, where each row contains the indices of the three
#     vertices of one of the M triangles in counterclockwise order.
#
def square(a,h0):
  # define the four vertices of the square
  points = [(0, 0), (a, 0), (a, a), (0, a)]
  # define the four edges between the four vertices of the square
  facets = round_trip_connect(0, len(points)-1)
  # initialize the mesh and set the vertices and the edges of the domain
  info = triangle.MeshInfo()
  info.set_points(points)
  info.set_facets(facets)
  # define a function that returns a boolean that is true if the triangle with
  # vertices vtx and area a has maximal edge length larger than the desired
  # maximal mesh width h0.
  def needs_refinement(vtx, a):
    return bool(max_edge_length(vtx) > h0)
  # create the mesh giving the mesh information info and the refinement
  # function needs_refinement as input
  mesh = triangle.build(info, refinement_func=needs_refinement)
  # read vertices and triangles of the mesh and convert the arrays to numpy
  # arrays
  p = np.array(mesh.points)
  t = np.array(mesh.elements)
  # return the vertices and triangles of the mesh
  return (p, t)

def show(p, t):
  codes = [Path.MOVETO,
           Path.LINETO,
           Path.LINETO,
           Path.CLOSEPOLY,]
  fig = plt.figure()
  ax = fig.add_subplot(111)
  [x_min, x_max, y_min, y_max] = [0, 0, 0, 0]
  for list in t:
      x_min = np.min([p[list[0]][0], p[list[1]][0], p[list[2]][0], x_min])
      x_max = np.max([p[list[0]][0], p[list[1]][0], p[list[2]][0], x_max])
      y_min = np.min([p[list[0]][1], p[list[1]][1], p[list[2]][1], y_min])
      y_max = np.max([p[list[0]][1], p[list[1]][1], p[list[2]][1], y_max])
      verts = [ p[list[0]], p[list[1]], p[list[2]], p[list[0]] ]
      path = Path(verts, codes)
      patch = patches.PathPatch(path, facecolor='none')
      ax.add_patch(patch)
  ax.set_xlim(x_min,x_max)
  ax.set_ylim(y_min,y_max)
  plt.show()

def grid_square(a,h0):
  vertical = np.sqrt((np.power(h0,2))/2)
  p = []
  m = int(a/vertical)
  [x,y] = [0,0]
  while y <= a:
      x = 0
      while x <= a:
          p.append([x,y])
          x = x + vertical
      y = y + vertical
  t = []
  for i in range(m):
      for j in range(m):
          [p1,p2,p3] = [i*(m+1)+j, i*(m+1)+j+1, (i+1)*(m+1)+j+1]
          t.append([p1,p2,p3])
          [p1,p2,p3] = [i*(m+1)+j, (i+1)*(m+1)+j+1, (i+1)*(m+1)+j]
          t.append([p1,p2,p3])
  return (p, t)

def max_mesh_width(p,t):
    maxi = 0
    for triangle in t:
        [p1, p2, p3] = [p[triangle[0]], p[triangle[1]], p[triangle[2]]]
        maxi = max(maxi, max_edge_length([p1, p2, p3]))
    return maxi

def circle(r,h0,n):
  # define the vertices on the circle
  points = []
  for i in range(n):
      x = r*np.cos(i*2*np.pi/n)
      y = r*np.sin(i*2*np.pi/n)
      points.append([x,y])
  # define the four edges between the four vertices of the square
  facets = round_trip_connect(0, len(points)-1)
  # initialize the mesh and set the vertices and the edges of the domain
  info = triangle.MeshInfo()
  info.set_points(points)
  info.set_facets(facets)
  # define a function that returns a boolean that is true if the triangle with
  # vertices vtx and area a has maximal edge length larger than the desired
  # maximal mesh width h0.
  def needs_refinement(vtx, a):
    return bool(max_edge_length(vtx) > h0)
  # create the mesh giving the mesh information info and the refinement
  # function needs_refinement as input
  mesh = triangle.build(info, refinement_func=needs_refinement)
  #mesh = triangle.build(info)
  # read vertices and triangles of the mesh and convert the arrays to numpy
  # arrays
  p = np.array(mesh.points)
  t = np.array(mesh.elements)
  # return the vertices and triangles of the mesh
  return (p, t)


