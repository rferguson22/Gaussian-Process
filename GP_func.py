import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import cdist

def GP(x_known, y_known, e_known,x_fit,lengths):

    '''
    Perform GP regression to find a predicted mean and uncertainty for unknown datapoints
    '''

    K = kernel_func(x_known, x_known,lengths) + e_known**2 * np.eye(len(e_known))
    K_s = kernel_func(x_known,x_fit,lengths)
    K_ss = kernel_func(x_fit,x_fit,lengths) 
    K_inv = inv(K)
        
    mu_s = K_s.T.dot(K_inv).dot(y_known)

    sigma_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, np.sqrt(abs(np.diag(sigma_s)))

#########################################################################

def kernel_func(x1,x2,l):

    '''
    Calculates the RBF kernel between 2 sets of points with given length scales for each dimension
    '''
    
    sq_norm=np.zeros((x1.shape[1],x2.shape[1]))

    for i in range(x1.shape[0]):
        sq_norm+=cdist(x1[i].reshape(-1,1),x2[i].reshape(-1,1),"sqeuclidean")/(l[i]**2)

    rbf = np.exp(-0.5*sq_norm)

    return rbf