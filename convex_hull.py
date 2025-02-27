import numpy as np
from scipy.spatial import ConvexHull

###########################################################################
def round_to_res(value,res):

    '''
    rounds the value to the nearest multiple of the given resolution
    '''    

    val=round(value/res)*res
    
    if res<1:
        val=np.around(val,str(res)[::-1].find('.'))
        
    return val

#################################################################################################

def sort_array(array):

    '''
    removes duplicate values and returns the sorted array
    '''

    return np.array(sorted(list(set(map(tuple,array)))))

#################################################################################################

def straight_line(pt1,pt2,step):
    
    '''
    Generates the points on the straight line between 2 points with a given step size
    '''

    d=np.argmax(abs((pt1-pt2)/step))
    
    if pt1[d]<pt2[d]:
        start=pt1
        end=pt2
    else:
        start=pt2
        end=pt1
    
    diff=end-start
    temp=np.zeros(len(start))
    
    for i in range(len(start)):
        temp[i]=start[i]
        
    line=[]
    line.append(start)


    while temp[d]<end[d]:
        temp[d]+=step[d]
        temp[d]=round_to_res(temp[d],step[d])
        t=(temp[d]-start[d])/diff[d]
        for i in range(len(temp)):
            if i!=d:
                temp[i]=(t*diff[i])+start[i]
                temp[i]=round_to_res(temp[i],step[i])
        line.append(temp.copy())

    return np.array(line)

###################################################################################

def boundary_edges(points,step):

    '''
    Generates the boundary edges of a polygon given a set of points
    '''

    i=1
    boundary=straight_line(points[0],points[1],step)
    

    while i<len(points):
        if i==len(points)-1:
            boundary_temp=straight_line(points[i],points[0],step)
        else:
            boundary_temp=straight_line(points[i],points[i+1],step)

        boundary=np.concatenate((boundary,boundary_temp))
        i+=1
    
    return sort_array(boundary)

########################################################################################

def remove_elt(array,index):

    '''
    Removes the array element at a specified dimension
    '''    

    return np.concatenate((array[:index],array[index+1:]))

#####################################################################################

def inc_step_func(step_size,dim,total_size):

    '''
    Creates an incremental step vector with given step size across a specified dimension
    '''

    inc=np.zeros(total_size)
    inc[dim]=step_size
    return inc

##########################################################################

def boundary_fill(inc_dim,edges,step):

    '''
    Fills in a polygon given a set of points
    '''

    inc_step=inc_step_func(step[inc_dim],inc_dim,len(edges.T))
    boundary=[]
    i=0
    while i<len(edges)-1:
        if (remove_elt(edges[i],inc_dim)==remove_elt(edges[i+1],inc_dim)).all():
            start=edges[i].copy()
            end=edges[i+1].copy()
            boundary.append(start.copy())
            while start[inc_dim]<end[inc_dim]:
                start+=inc_step
                for j in range(len(start)):
                    start[j]=round_to_res(start[j],step[j])
                boundary.append(start.copy())
            boundary.append(end.copy())

        i+=1

    boundary.append(edges[-1])
    
    boundary_temp=np.concatenate((np.array(boundary),edges))
    
    return sort_array(boundary_temp)

#########################################################################

def hull_boundary(vertices,step):
    
    '''
    Finds the boundary of the convex hull, given its vertices and a step size
    '''    

    edges=boundary_edges(vertices,step)
    
    inc_dim=len(vertices)-1
    
    while (max(vertices[:,inc_dim])-min(vertices[:,inc_dim]))==0:
        inc_dim-=1

    boundary=boundary_fill(inc_dim,edges,step)
    
    return boundary  

#####################################################################
     
def hull_vertices(points,vertices_index):

    '''
    Finds the coordinates of the vertices of the convex hull
    '''

    vertices=[]
    
    for i in range(len(vertices_index)):
        vertices.append(points[vertices_index[i]])
        
    return np.array(vertices)

##################################################################

def hull_fill_in(total_boundary,step):

    '''
    Fills in the boundary of the convex hull and reorders the step size for interpolation
    '''    

    hull_points=boundary_fill(len(total_boundary.T)-1,total_boundary,step)
    
    hull_points=sort_array(np.concatenate((hull_points[:,1:].T,hull_points[:,:1].T)).T)
    
    step=np.concatenate((step[1:],step[:1]))
    
    return hull_points,step

#########################################################

def fill_convex_hull(points,step):
  
    '''
    Fills in the convex hull by finding the boundaries and interpolating given step sizes for each dimension
    '''    

    hull=ConvexHull(points)
    
    total_boundary=hull_boundary(hull_vertices(points,hull.simplices[0]),step)
    
    j=1
    while j<len(hull.simplices):
        boundary_temp=hull_boundary(hull_vertices(points,hull.simplices[j]),step)
        total_boundary=np.concatenate((total_boundary,boundary_temp))
        j+=1
    
    hull_points=sort_array(total_boundary)

    for i in range(len(hull_points.T)):
        hull_points,step=hull_fill_in(hull_points,step)
    
    return hull_points
