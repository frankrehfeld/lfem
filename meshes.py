import meshpy.triangle as triangle
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

#
# show(p,t)
#
# plots a mesh given by points p and triangles t
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
#
def show(p, t):
    bn = boundaryNodes(p,t)
    In = interiorNodes(p,t)
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
    for point in bn:
        ax.plot(p[point][0],p[point][1], 'ro')
    for point in In:
        ax.plot(p[point][0],p[point][1], 'bo')
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    plt.show()

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

#
# grid_square(a,h0)
#
# Function that produces an structured mesh of a square. The square has side
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
def grid_square(a,h0):
    vertical = np.sqrt((np.power(h0,2))/2)
    p = np.ndarray(shape=[0, 2])
    m = int(a/vertical)
    [x,y] = [0,0]
    while y <= a:
        x = 0
        while x <= a:
            p = np.append(p, [[x,y]], axis=0)
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

#
# max_mesh_width(p,t)
#
# computes the maximal mesh width, which is the length of the longest
# edge of a triangle in the mesh.
#
# input:
# p - Numpy-array of size Nx2 with the x- and y-coordinates of the N vertices
#     in the mesh
# t - Numpy-array of size Mx3, where each row contains the indices of the three
#     vertices of one of the M triangles in counterclockwise order.
#
# output:
# maxi - maximal mesh width of the mesh spanned by p and t
#
def max_mesh_width(p,t):
    maxi = 0
    for triangle in t:
        [p1, p2, p3] = [p[triangle[0]], p[triangle[1]], p[triangle[2]]]
        maxi = max(maxi, max_edge_length([p1, p2, p3]))
    return maxi

#
# circle(r,h0,n)
#
# Function that produces an unstructured mesh of a circle. The circle has
# radius r and its center is positioned at the origin. The maximal mesh
# width of the produced mesh is smaller or equal to h0. (The maximal mesh
# width of a mesh is the maximal distance of two adjacent vertices, or
# equivalenty, the maximal length of all edges in the mesh.) There are n
# points positioned on the boundary of the circle.
#
# input:
# r  - radius of the circle
# h0 - upper bound for maximal mesh width
# n  - amount of points positioned on the boundary
#
# output:
# p - Numpy-array of size Nx2 with the x- and y-coordinates of the N vertices
#     in the mesh
# t - Numpy-array of size Mx3, where each row contains the indices of the three
#     vertices of one of the M triangles in counterclockwise order.
#
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
