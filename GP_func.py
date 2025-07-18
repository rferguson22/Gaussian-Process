import numpy as np
from numpy.linalg import solve,cholesky

def GP(x_known, y_known, e_known, x_fit, lengths, batch_size=10000):

    '''
    Perform GP regression to find a predicted mean and uncertainty for unknown datapoints
    '''

    y_fit = []
    e_fit = []

    K = kernel_func(x_known, x_known, lengths) + np.diag(e_known**2)
    L = cholesky(K)  

    alpha = solve(L.T, solve(L, y_known))

    total_points = x_fit.shape[1]
    
    for start in range(0, total_points, batch_size):
        end = min(start + batch_size, total_points)
        x_batch = x_fit[:, start:end]

        K_s = kernel_func(x_known, x_batch, lengths)

        K_ss_diag=np.ones(x_batch.shape[1])

        mu_s = K_s.T @ alpha

        v = solve(L, K_s)
        var_s = np.clip(K_ss_diag - np.sum(v**2, axis=0), 1e-12, None)
        sigma_s = np.sqrt(var_s)

        y_fit.append(mu_s)
        e_fit.append(sigma_s)

    return np.concatenate(y_fit), np.concatenate(e_fit)

#########################################################################

def kernel_func(x1,x2,l):

    '''
    Calculates the RBF kernel between 2 sets of points with given length scales for each dimension
    '''
    

    x1_scaled=x1/l[:,None]
    x2_scaled=x2/l[:,None]

    x1_sq = np.sum(x1_scaled**2, axis=0).reshape(-1, 1)
    x2_sq = np.sum(x2_scaled**2, axis=0).reshape(1, -1)

    sq_dist = x1_sq + x2_sq - 2 * np.dot(x1_scaled.T, x2_scaled)
    sq_dist = np.maximum(sq_dist, 0)  

    return np.exp(-0.5*sq_dist)