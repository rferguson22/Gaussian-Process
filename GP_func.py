import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import cdist

def GP(xy_known, z_known, e_known,xy,lengths):

    '''
    Perform GP regression to find a predicted mean and uncertainty for unknown datapoints
    '''

    K = kernel_func(xy_known, xy_known,lengths) + e_known**2 * np.eye(len(e_known))
    K_s = kernel_func(xy_known,xy,lengths)
    K_ss = kernel_func(xy,xy,lengths) 
    K_inv = inv(K)
        
    mu_s = K_s.T.dot(K_inv).dot(z_known)

    sigma_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, np.sqrt(abs(np.diag(sigma_s)))

#########################################################################

def kernel_func(xy1,xy2,l):

    '''
    Calculates the RBF kernel between 2 sets of points with given length scales for each dimension
    '''
    
    sq_norm=np.zeros((xy1.shape[1],xy2.shape[1]))

    for i in range(xy1.shape[0]):
        sq_norm+=cdist(xy1[i].reshape(-1,1),xy2[i].reshape(-1,1),"sqeuclidean")/(l[i]**2)

    rbf = np.exp(-0.5*sq_norm)

    return rbf