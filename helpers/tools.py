import numpy as np

def center_of_mass(matrix:np.ndarray, x:float, y:float)-> tuple:
    """Returns a center of mass coordinates.

    Args:
        matrix (np.ndarray): matrix with mass or intensity values
        x (float): x coordinate vectors (around peak)
        y (float): y coordinate vectors (around peak)

    Returns:
        (x,y) (tuple): coordinates of the center of mass
    """

    size=matrix.shape
    if size[0]!=len(x) and size[1]!=len(y):
        print('Error: width is not matching matrix size!')
    else:        
        Y,X=np.meshgrid(y,x)
        mass=np.nansum(np.nansum(matrix))
        xpos=np.nansum(np.nansum(matrix*X))/mass
        ypos=np.nansum(np.nansum(matrix*Y))/mass
    return (xpos, ypos)
    

def arrayIndex(M,ind):
    #Array and indices are numpy arrays. If you want to adress elements Matlab
    #like with an array. New array is an array of elements with indices.
    new_array=[]
    for j in range(0,len(ind)):
        new_array=np.append(new_array,M[ind[j]])
    return new_array
