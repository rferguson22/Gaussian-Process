import numpy as np
from numpy.linalg import solve,cholesky,inv

def GP1(x_known, y_known, e_known, x_fit, lengths, batch_size=10000):

    '''
    Perform GP regression to find a predicted mean and uncertainty for unknown datapoints
    '''

    y_fit = []
    e_fit = []

    K = kernel_func(x_known, x_known, lengths) + np.diag(e_known**2)

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError as err:
        print("Cholesky failed: matrix not PD.")
        print("Min eigenvalue:", np.min(np.linalg.eigvalsh(K)))
        print("Condition number:", np.linalg.cond(K))
        print("Lengths used:", lengths)
        print("e_known min/max:", np.min(e_known), np.max(e_known))
        print("x_known shape:", x_known.shape)
        print("K diagonal min/max:", np.min(np.diag(K)), np.max(np.diag(K)))
        print("K:\n", K)
        raise


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

def GP(x_known, y_known, e_known,x_fit,lengths):

    '''
    Perform GP regression to find a predicted mean and uncertainty for unknown datapoints
    '''

    y_fit=np.array([])
    e_fit=np.array([])

    starting_val=10000
    l=0

    K = kernel_func(x_known, x_known,lengths) + e_known**2 * np.eye(len(e_known))
    K_inv = inv(K)

    while l<len(x_fit.T):

        l+=starting_val

        if l>len(x_fit.T):
            x_fit_temp=x_fit.T[l-starting_val:].T
        else:
            x_fit_temp=x_fit.T[l-starting_val:l].T


        K_s = kernel_func(x_known,x_fit_temp,lengths)
        K_ss = kernel_func(x_fit_temp,x_fit_temp,lengths) 
            
        mu_s = K_s.T.dot(K_inv).dot(y_known)

        sigma_s = np.sqrt(abs(np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))))

        y_fit=np.concatenate((y_fit,mu_s))
        e_fit=np.concatenate((e_fit,sigma_s))
    
    return y_fit,e_fit

####################################################################################################################

def kernel_func(x1,x2,l):

    '''
    Calculates the RBF kernel between 2 sets of points with given length scales for each dimension
    '''
    
    #amp, ls = l[0], l[1:]
    x1_scaled = x1 / l[:, None]
    x2_scaled = x2 / l[:, None]

    x1_sq = np.sum(x1_scaled**2, axis=0).reshape(-1, 1)
    x2_sq = np.sum(x2_scaled**2, axis=0).reshape(1, -1)

    sq_dist = x1_sq + x2_sq - 2 * np.dot(x1_scaled.T, x2_scaled)
    sq_dist = np.maximum(sq_dist, 0)  

    #return amp * np.exp(-0.5 * sq_dist)
    return np.exp(-0.5 * sq_dist)

###########################################################################################################

def GP_with_inverse(x_known, y_known, e_known, x_fit, lengths, batch_size=10000):
    """
    GP regression that also returns K_inv for vectorized LOO computations.
    """

    # Compute covariance matrix with noise
    K = kernel_func(x_known, x_known, lengths) + np.diag(e_known**2)

    # Cholesky decomposition for numerical stability
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError as err:
        print("Cholesky failed: matrix not PD.")
        print("Min eigenvalue:", np.min(np.linalg.eigvalsh(K)))
        print("Condition number:", np.linalg.cond(K))
        print("Lengths used:", lengths)
        raise

    # Solve for alpha = K^-1 y
    alpha = solve(L.T, solve(L, y_known))

    # Compute K_inv using Cholesky (more stable than direct inversion)
    identity = np.eye(K.shape[0])
    K_inv = solve(L.T, solve(L, identity))

    # Predict at x_fit in batches
    y_fit_list = []
    e_fit_list = []
    total_points = x_fit.shape[1]

    for start in range(0, total_points, batch_size):
        end = min(start + batch_size, total_points)
        x_batch = x_fit[:, start:end]

        K_s = kernel_func(x_known, x_batch, lengths)
        K_ss_diag = np.ones(x_batch.shape[1])

        # Predictive mean
        mu_s = K_s.T @ alpha

        # Predictive variance
        v = solve(L, K_s)
        var_s = np.clip(K_ss_diag - np.sum(v**2, axis=0), 1e-12, None)
        sigma_s = np.sqrt(var_s)

        y_fit_list.append(mu_s)
        e_fit_list.append(sigma_s)

    y_fit = np.concatenate(y_fit_list)
    e_fit = np.concatenate(e_fit_list)

    return y_fit, e_fit, K_inv
