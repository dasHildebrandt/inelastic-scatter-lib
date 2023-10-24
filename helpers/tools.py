import numpy as np

def center_of_mass(matrix:np.ndarray, x:float, y:float):
    """Input matrix in np.array format. x and y are the vectors which determine 
    the area of interest around the peak."""

    size=matrix.shape
    if size[0]!=len(x) and size[1]!=len(y):
        print('Error: width is not matching matrix size!')
    else:        
        Y,X=np.meshgrid(y,x)
        #Grids must match image and positions.
        mass=np.nansum(np.nansum(matrix))
        xpos=np.nansum(np.nansum(matrix*X))/mass
        ypos=np.nansum(np.nansum(matrix*Y))/mass
    return xpos, ypos
    

def arrayIndex(M,ind):
    #Array and indices are numpy arrays. If you want to adress elements Matlab
    #like with an array. New array is an array of elements with indices.
    new_array=[]
    for j in range(0,len(ind)):
        new_array=np.append(new_array,M[ind[j]])
    return new_array
