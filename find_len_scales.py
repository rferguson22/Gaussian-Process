import os
import numpy as np
from numpy.linalg import inv,cholesky,solve
import math

from smt.sampling_methods import LHS
import emcee
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import corner
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.stats import gaussian_kde,kstest,ks_2samp
from scipy.signal import find_peaks
from scipy.optimize import minimize,differential_evolution
from scipy.stats import norm
from scipy.special import expit
from scipy.linalg import solve

from sklearn.mixture import GaussianMixture

from GP_func import GP,GP1,GP_with_inverse,kernel_func

###################################################################################################

def sigma_check_loo_smooth(lengths, x_known, y_known, e_known, sharpness=20.0):

    def kernel_func(x1, x2, l):
        x1_scaled = x1 / l[:, None]
        x2_scaled = x2 / l[:, None]
        x1_sq = np.sum(x1_scaled**2, axis=0).reshape(-1, 1)
        x2_sq = np.sum(x2_scaled**2, axis=0).reshape(1, -1)
        sq_dist = x1_sq + x2_sq - 2 * np.dot(x1_scaled.T, x2_scaled)
        sq_dist = np.maximum(sq_dist, 0)
        return np.exp(-0.5 * sq_dist)

    def gp_loo_predictions(x_known, y_known, e_known, lengths):
        K = kernel_func(x_known, x_known, lengths) + np.diag(e_known**2)
        K_inv = np.linalg.inv(K)
        alpha = K_inv @ y_known
        loo_var = 1.0 / np.diag(K_inv)
        loo_mean = y_known - alpha / np.diag(K_inv)
        loo_std = np.sqrt(np.maximum(loo_var, 1e-12))
        return loo_mean, loo_std

    def smooth_indicator(z, sharpness=20.0):
        return expit(sharpness * (1.0 - np.abs(z)))

    # Reject invalid lengths
    if np.any(lengths <= 0):
        return -1e13, None, None, None

    y_pred, e_pred = gp_loo_predictions(x_known, y_known, e_known, lengths)

    # Sigma grid
    sigma_vals = np.linspace(0, 3, 1000)
    expected_percents = sigma_to_percent(sigma_vals)

    scaled_e = e_pred[:, None] * sigma_vals[None, :]
    pulls = (y_pred[:, None] - y_known[:, None]) / np.maximum(scaled_e, 1e-12)

    fractions_within = np.mean(smooth_indicator(pulls, sharpness=sharpness), axis=0)
    loss = -np.sum((fractions_within - expected_percents)**2)

    return loss, sigma_vals, fractions_within, expected_percents

####################################################################################################

def sigma_check_smooth_wrapper(lengths, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds, upper_bounds):
    if np.any(lengths <= lower_bounds) or np.any(lengths >= upper_bounds):
        return -1e13
    loss, _, _, _ = sigma_check_loo_smooth(lengths, x_known, y_known, e_known, sharpness=20.0)
    return loss

##################################################################################################

def sigma_check_old(lengths,x_known,y_known,e_known,sigma,expected_vals,lower_bounds,upper_bounds):

    '''
    Calculates the loss function for a given length scale
    '''

    if np.any(lengths <= lower_bounds) or np.any(lengths>=upper_bounds):
        return -10e12

    y_fit,e_fit=GP1(x_known,y_known,e_known,x_known,lengths)
    
    total=0

    for i in range(len(sigma)):
        percent = sigma_to_percent(sigma[i])
        total-=abs(calculate_std_percent(y_fit,y_known,e_fit,sigma[i])-percent)
    
    return total

###################################################################################################

def sigma_check(lengths, x_known, y_known, e_known,sigma_vals,expected_percents,lower_bounds,upper_bounds):

    '''
    Computes the loss function:
    '''

    if np.any(lengths<= lower_bounds) or np.any(lengths>=upper_bounds):
    #if np.any(lengths<=0):
        return -1e13

    y_fit, e_fit = GP(x_known, y_known, e_known, x_known, lengths)
    
    scaled_e = e_fit[:, None] * sigma_vals[None, :] 

    #total_uncertainty = np.sqrt((scaled_e)**2 + e_known[:, None]**2)

    #pulls = (y_fit[:, None] - y_known[:, None]) / np.maximum(total_uncertainty, 1e-12)

    pulls = (y_fit[:, None] - y_known[:, None]) / np.maximum(scaled_e, 1e-12)

    fractions_within = np.mean(np.abs(pulls) <= 1, axis=0)

    loss = -np.sum(np.abs(fractions_within - expected_percents))

    return loss

##########################################################################################

def sigma_check_loo(lengths, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds, upper_bounds):
    """
    LOO loss:
    - Predict each point from the GP trained on all others
    - Count containment fractions in sigma bands
    - Compare to expected percents
    - Return sum of differences
    """

    if np.any(lengths <= lower_bounds) or np.any(lengths >= upper_bounds):
        return -1e13

    # GP inverse once (needed for fast LOO)
    _, _, K_inv = GP_with_inverse(x_known, y_known, e_known, x_known, lengths)

    # Vectorised LOO predictions
    diag_Kinv = np.diag(K_inv)
    y_loo = y_known - (K_inv @ y_known) / diag_Kinv
    e_loo = 1.0 / np.sqrt(diag_Kinv)

    scaled_e = e_loo[:, None] * sigma_vals[None, :]
    pulls = (y_loo[:, None] - y_known[:, None]) / np.maximum(scaled_e, 1e-12)

    within = (np.abs(pulls) <= 1).astype(float)  # shape (n_points, n_sigma)

    fractions_within = np.mean(within, axis=0)  # shape (n_sigma,)

    loss = -np.sum(np.abs(fractions_within - expected_percents))

    return loss

##############################################################################################################

def sigma_check2(lengths, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds, upper_bounds):
    """
    Computes the loss function using leave-one-out cross-validation,
    summing the differences across all left-out points.
    Optimized to avoid repeated copying of arrays.
    """

    if np.any(lengths[1:] <= 0):
        return -1e13

    n_points = len(y_known)
    n_sigma = len(sigma_vals)
    fractions_within_total = np.zeros(n_sigma)

    # Precompute mask indices for leave-one-out
    indices = np.arange(n_points)

    for i in range(n_points):
        mask = indices != i  # boolean mask to leave out point i

        # Fit GP on all points except i
        y_pred, e_pred = GP(
            x_known[mask], y_known[mask], e_known[mask],
            x_known[i:i+1], lengths
        )  # predict only left-out point

        scaled_e = e_pred[:, None] * sigma_vals[None, :]

        # Compute pull for the left-out point
        pulls = (y_pred[:, None] - y_known[i]) / np.maximum(scaled_e, 1e-12)

        # Accumulate sum of fractions within 1 sigma
        fractions_within_total += (np.abs(pulls) <= 1).astype(float).flatten()

    # Loss: sum of absolute differences from expected percentages
    loss = -np.sum(np.abs(fractions_within_total - expected_percents))

    return loss

######################################################################################################

def sigma_check_sigmoid(lengths, x_known, y_known, e_known,
                       sigma_vals, expected_percents,
                       lower_bounds, upper_bounds, a=20.0):
    
    """
    Smooth coverage loss for GP hyperparameter optimization.
    
    lengths: kernel hyperparameters
    x_known, y_known, e_known: training data and observation noise
    sigma_vals: array of sigma thresholds to check (e.g., 1000 values from 0-3)
    expected_percents: theoretical coverage fractions for each threshold
    lower_bounds, upper_bounds: bounds for hyperparameters
    a: sigmoid steepness
    """
    
    # Reject invalid hyperparameters
    if np.any(lengths <= lower_bounds) or np.any(lengths >= upper_bounds):
        return -1e13

    # GP predictions
    y_fit, e_fit = GP(x_known, y_known, e_known, x_known, lengths)  # e_fit includes noise

    # Standardised residuals
    z = (y_fit[:, None] - y_known[:, None]) / np.maximum(e_fit[:, None] * sigma_vals[None, :], 1e-12)

    # Smooth indicator using sigmoid
    S = expit(a * (1.0 - np.abs(z)))   # approx 1_{|z| <= 1}

    # Empirical coverage at each sigma threshold
    fractions_within = S.mean(axis=0)

    # Smooth loss: mean squared difference from expected coverage
    loss = -np.sum((fractions_within - expected_percents) ** 2)

    return loss

#####################################################################################################

def sigma_to_percent(x):

    '''
    Calculates the percentage of points within x*standard deviation
    '''

    upper=norm.cdf(x)
    lower=norm.cdf(-x)

    return upper-lower

#######################################################################################################

def ks_loss1(lengths, x_known, y_known, e_known, sigma_vals, lower_bounds,upper_bounds):
    '''
    Computes the loss using a Kolmogorov-Smirnov test
    between the normalised residuals ("pulls") and
    a normal distribution with variance = 1/sigma^2.
    '''

    if np.any(lengths[1:] <= lower_bounds) or np.any(lengths[1:]>=upper_bounds):
        return -1e13

    y_fit, e_fit = GP(x_known, y_known, e_known, x_known, lengths)
    
    scaled_e = e_fit[:, None] * sigma_vals[None, :]  # shape (N_samples, N_sigma)
    pulls = (y_fit[:, None] - y_known[:, None]) / np.maximum(scaled_e, 1e-12)  # same shape
    
    N = pulls.shape[0]
    pulls_sorted = np.sort(pulls, axis=0)  # sort pulls column-wise
    
    # Empirical CDF values for KS test
    i = np.arange(1, N + 1)[:, None]  # shape (N_samples, 1)
    F_n1 = i / N
    F_n2 = (i - 1) / N

    # Theoretical CDF under H0 for each sigma:
    sigma_vals_reshaped = sigma_vals[None, :]  # (1, N_sigma)
    F_theoretical = norm.cdf(sigma_vals_reshaped * pulls_sorted)  # (N_samples, N_sigma)

    D1 = np.abs(F_n1 - F_theoretical)
    D2 = np.abs(F_n2 - F_theoretical)
    D = np.maximum(D1, D2)

    ks_stats = np.max(D, axis=0)  # max difference per sigma
    
    loss = -np.sum(ks_stats)
    return loss

##################################################################################################################

def convert_to_cdf(percent_within):

    return 0.5 * ((percent_within / 100) + 1)

##############################################################################################################

def ks_loss(lengths,x_known,y_known,e_known,sigma_vals,expected_percents,lower_bounds,upper_bounds):

    if np.any(lengths <= lower_bounds) or np.any(lengths >= upper_bounds):
    #if np.any(lengths<=0):
        return -1e13

    y_fit, e_fit = GP1(x_known, y_known, e_known, x_known, lengths)
    
    scaled_e = e_fit[:, None] * sigma_vals[None, :] 

    pulls = (y_fit[:, None] - y_known[:, None]) / np.maximum(scaled_e, 1e-12)

    measured_percents = np.mean(np.abs(pulls) <= 1, axis=0)

    #measured_cdf=convert_to_cdf(measured_percents)

    D, _ = ks_2samp(expected_percents, measured_percents)
    
    return -D

##############################################################################################################

def len_scale_ks(x_known, y_known, e_known, MC_progress, MC_plotting, labels, out_file_name):

    '''
    Finds the optimal length scale based on the KS-test loss function,
    using MCMC for exploration and deterministic selection of the best candidate.
    '''
    original_file_path = Path(out_file_name)
    plotting_path = original_file_path.parent

    ndim = len(x_known)
    nwalkers = 40 * ndim
    max_n = 2000 * ndim

    r_hat_tol = 1.18
    tau_tol = 0.15

    ranges = np.abs(np.max(x_known, axis=1) - np.min(x_known, axis=1))
    lower_bounds = 0.01 * ranges/len(x_known.T)
    upper_bounds = ranges
    endpoints = np.column_stack((lower_bounds, upper_bounds))

    sampling = LHS(xlimits=np.array(endpoints), criterion='center')
    initial_positions = sampling(nwalkers)
    initial_positions += 1e-5 * np.random.randn(*initial_positions.shape)

    filename = "backend.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    print("Beginning MCMC")

    sigma_vals = np.linspace(0.001, 3, 1000) 
    expected_percents = sigma_to_percent(sigma_vals)
    #expected_cdf=convert_to_cdf(expected_percents)


    num_cores = max(1, os.cpu_count() // 4)
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=num_cores) as pool:
        index = 0
        autocorr = np.empty(max_n)
        r_hat_conv = False
        
        old_tau = np.inf
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, ks_loss,
            args=(x_known,y_known,e_known,sigma_vals,expected_percents,lower_bounds, upper_bounds),
            backend=backend,
            pool=pool
        )
        for sample in sampler.sample(initial_positions, iterations=max_n, progress=MC_progress):
            if sampler.iteration % 100 != 0:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            chains = sampler.get_chain(discard=50, thin=5, flat=False)
            if chains.shape[1] > 1:
                r_hat = calc_r_hat(chains)
                r_hat_conv = np.all(r_hat < r_hat_tol)
                if MC_progress:
                    print("r_hat:\t\t" + str(r_hat))

            tau_conv = np.all((np.abs(old_tau - tau) / tau) < tau_tol)
            if MC_progress:
                print("tau_stability:\t" + str(np.abs(old_tau - tau) / tau))

            if r_hat_conv and tau_conv:
                break
            old_tau = tau

    burnin = int(0.5 * sampler.iteration)
    samples = sampler.get_chain(discard=burnin, thin=10, flat=True)

    print("MCMC converged.")

    
    num_peaks, x, density = test_unimode(samples, dim=0)

    print("MCMC converged. Checking for multimodal surface")

    if MC_plotting:
        fig = corner.corner(samples, labels=labels[:-2])
        fig.savefig(plotting_path / "GP_corner_plot.png")

        plt.figure(figsize=(8, 6))
        plt.plot(x, density, label="KDE")
        plt.title(f"KDE and Peak Detection (Peaks Found: {num_peaks})")
        plt.xlabel("Dim 1")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(plotting_path / "GP_KDE_plots.png")
        plt.show()

    if num_peaks > 1:
        print("Multimodal distribution detected. Performing Clustering")

        silhouette_scores = []
        K_values = range(2, 11)

        for K in K_values:
            kmeans = KMeans(n_clusters=K, random_state=0) 
            cluster_labels = kmeans.fit_predict(samples)
            score = silhouette_score(samples, cluster_labels)
            silhouette_scores.append(score)

        if MC_plotting:
            plt.figure(figsize=(8, 6))
            plt.plot(K_values, silhouette_scores, "-o", color="blue")
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Score for Optimal K")
            plt.grid(True)
            plt.savefig(plotting_path / "GP_silhouette_scores.png")
            plt.show()

        optimal_K = K_values[np.argmax(silhouette_scores)]
        print(f"Optimal number of Clusters (K): {optimal_K}")

        kmeans = KMeans(n_clusters=optimal_K, random_state=0)
        cluster_labels = kmeans.fit_predict(samples)
        modes = kmeans.cluster_centers_

    else:
        print("Single mode surface")
        modes = np.mean(samples, axis=0, keepdims=True)

    print("Modes of surface:")
    print(modes)
    


    def func_minimise(lengths):
        return -ks_loss(lengths,x_known,y_known,e_known,sigma_vals,expected_percents,lower_bounds, upper_bounds)    

    score=1e12
    best=[]
    
    for j in range(len(modes)):
        result=minimize(func_minimise,modes[j],method="Nelder-Mead")
        if result.fun<score:
            best=result.x
            score=result.fun

    print("Optimal length scale found:")
    print(best)
    print("Value of loss function:")
    print(score)

    if os.path.exists(filename):
        os.remove(filename)

    return best, score

###########################################################################################################

def bounds(X, lower_frac=0.1, upper_frac=10.0):

    """
    Compute sensible lower and upper bounds for RBF kernel length scales
    using the median pairwise distance heuristic (vectorised).

    Parameters
    ----------
    X : array-like of shape (ndims, nsamples)
        Input data (features along rows, samples along columns).
    lower_frac : float, default=0.1
        Fraction of the median distance to use as the lower bound.
    upper_frac : float, default=10.0
        Multiple of the median distance to use as the upper bound.

    Returns
    -------
    bounds : np.ndarray of shape (ndims, 2)
        Array of (lower_bound, upper_bound) for each dimension.
    """
    X = np.asarray(X)  # (ndims, nsamples)
    ndims, nsamples = X.shape

    # Pairwise absolute differences: (ndims, nsamples, nsamples)
    diffs = np.abs(X[:, :, None] - X[:, None, :])

    # Set diagonals to NaN to ignore self-distances
    idx = np.arange(nsamples)
    diffs[:, idx, idx] = np.nan

    # Median per dimension (ignoring NaNs)
    median_dist = np.nanmedian(diffs, axis=(1, 2))

    # Compute bounds
    lower = lower_frac * median_dist
    upper = upper_frac * median_dist

    return np.stack([lower, upper], axis=1),lower,upper

##########################################################################################################

def bounds1(X, lower_frac=0.05, upper_frac=1.0):

    """
    Compute sensible lower and upper bounds for RBF kernel length scales.
    
    Parameters:
        X : array-like of shape (ndims, nsamples)
            Input data.
        lower_frac : float
            Fraction of minimum spacing to use as lower bound (default 0.05).
        upper_frac : float
            Fraction of data range to use as upper bound (default 2.0).

    Returns:
        bounds : np.ndarray of shape (ndims, 2)
            Array of (lower_bound, upper_bound) for each dimension.
    """
    X = np.asarray(X)
    ndims, nsamples = X.shape

    # Compute pairwise distances for all dimensions
    # Expand dims: X[:, :, None] - X[:, None, :] gives shape (ndims, nsamples, nsamples)
    diffs = np.abs(X[:, :, None] - X[:, None, :])

    # Set diagonal to np.inf to ignore zero distances
    np.fill_diagonal(diffs.reshape(-1, nsamples), np.inf)

    # Minimum non-zero distance per dimension
    min_dist = np.min(diffs, axis=(1, 2))

    # Data range per dimension
    data_range = X.max(axis=1) - X.min(axis=1)

    lower = lower_frac * min_dist
    upper = upper_frac * data_range

    endpoints = np.stack([lower, upper], axis=1)

    return endpoints,lower,upper

##########################################################################################################

def log_prob_fn(params, x_known, y_known, e_known, sigma_vals, expected_percents, lengths_lower, lengths_upper):

    """
    MCMC log-probability using the full hyperparameter vector.
    """
    n_features = x_known.shape[0]
    lengths = params[:n_features]
    w_rbf = params[n_features]
    w_linear = params[n_features + 1]
    w_poly = params[n_features + 2:]

    # enforce bounds
    if np.any(lengths <= lengths_lower) or np.any(lengths >= lengths_upper):
        return -1e13
    if np.any(np.array([w_rbf, w_linear] + list(w_poly)) < 0):
        return -1e13

    return -sigma_check(params, x_known, y_known, e_known, sigma_vals, expected_percents)


##########################################################################################################

def len_scale_sigma(x_known, y_known, e_known, MC_progress, MC_plotting, labels, out_file_name):
    
    '''
    Finds the optimal length scale based on the KS-test loss function,
    using MCMC for exploration and deterministic selection of the best candidate.
    '''
    original_file_path = Path(out_file_name)
    plotting_path = original_file_path.parent

    ndim = 1
    nwalkers = 40* ndim
    max_n = 2000 * ndim

    r_hat_tol = 1.1
    tau_tol = 0.1

    x_known = np.atleast_2d(x_known)  # ensures x_known has shape (1, N) if it was 1D
    ranges = np.max(x_known, axis=1) - np.min(x_known, axis=1)
    lower_bounds = 0.01 * ranges / x_known.shape[1]  # (1,)
    upper_bounds = ranges                            # (1,)
    endpoints = np.column_stack((lower_bounds, upper_bounds))  # (1,2)

    sampling = LHS(xlimits=endpoints, criterion='center')
    initial_positions = sampling(nwalkers)  # shape (40,1)


    #endpoints,lower_bounds,upper_bounds = bounds(x_known, 0.5,2)

    '''
    endpoints=[]
    for i in range(len(x_known)):
        diff=(max(x_known[i])-min(x_known[i]))/len(x_known[i].T)
        endpoints.append([1e-16,10*diff])
    '''

    
    sampling = LHS(xlimits=np.array(endpoints), criterion='center')
    initial_positions = sampling(nwalkers)
    initial_positions += 1e-5 * np.random.randn(*initial_positions.shape)


    filename = "backend.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    print("Beginning MCMC")

    sigma_vals = np.linspace(0.001, 3, 1000)
    expected_percents = sigma_to_percent(sigma_vals)

    num_cores = max(1, os.cpu_count() // 4)
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=num_cores) as pool:
        index = 0
        autocorr = np.empty(max_n)
        r_hat_conv = False
        
        old_tau = np.inf
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, sigma_check,
            args=(x_known, y_known, e_known,sigma_vals,expected_percents,lower_bounds,upper_bounds),
            backend=backend,
            pool=pool
        )
        for sample in sampler.sample(initial_positions, iterations=max_n, progress=MC_progress):
            if sampler.iteration % 100 != 0:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            chains = sampler.get_chain(discard=50, thin=5, flat=False)
            if chains.shape[1] > 1:
                r_hat = calc_r_hat(chains)
                r_hat_conv = np.all(r_hat < r_hat_tol)
                if MC_progress:
                    print("r_hat:\t\t" + str(r_hat))

            tau_conv = np.all((np.abs(old_tau - tau) / tau) < tau_tol)
            if MC_progress:
                print("tau_stability:\t" + str(np.abs(old_tau - tau) / tau))

            if r_hat_conv and tau_conv:
                break
            old_tau = tau

    burnin = int(0.4 * sampler.iteration)
    samples = sampler.get_chain(discard=burnin, thin=10, flat=True)

    print("MCMC converged.")

    num_peaks, x, density = test_unimode(samples, dim=0)

    print("Checking for multimodal surface")

    if MC_plotting:
        fig = corner.corner(samples, labels=labels[:-2])
        fig.savefig(plotting_path / "GP_corner_plot.png")

        plt.figure(figsize=(8, 6))
        plt.plot(x, density, label="KDE")
        plt.title(f"KDE and Peak Detection (Peaks Found: {num_peaks})")
        plt.xlabel("Dim 1")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(plotting_path / "GP_KDE_plots.png")
        plt.show()

    if num_peaks > 1:
        print("Multimodal distribution detected. Performing Clustering")

        silhouette_scores = []
        K_values = range(2, 11)

        for K in K_values:
            kmeans = KMeans(n_clusters=K, random_state=0) 
            cluster_labels = kmeans.fit_predict(samples)
            score = silhouette_score(samples, cluster_labels)
            silhouette_scores.append(score)

        if MC_plotting:
            plt.figure(figsize=(8, 6))
            plt.plot(K_values, silhouette_scores, "-o", color="blue")
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Score for Optimal K")
            plt.grid(True)
            plt.savefig(plotting_path / "GP_silhouette_scores.png")
            plt.show()

        optimal_K = K_values[np.argmax(silhouette_scores)]
        print(f"Optimal number of Clusters (K): {optimal_K}")

        kmeans = KMeans(n_clusters=optimal_K, random_state=0)
        cluster_labels = kmeans.fit_predict(samples)
        modes = kmeans.cluster_centers_

    else:
        print("Single mode surface")
        modes = np.mean(samples, axis=0, keepdims=True)

    print("Modes of surface:")
    print(modes)



    def func_minimise(lengths):
        return -sigma_check(lengths,x_known,y_known,e_known,sigma_vals,expected_percents,lower_bounds,upper_bounds)  

    score=1e12
    best=[]
    
    for j in range(len(modes)):
        result=minimize(func_minimise,modes[j],method="Nelder-Mead")
        if result.fun<score:
            best=result.x
            score=result.fun


    print("Optimal length scale found:")
    print(best)
    print("Value of loss function:")
    print(score)

    if os.path.exists(filename):
        os.remove(filename)

    return best, score

###########################################################################################################

def test_unimode(samples, dim=0):
        
        kde = gaussian_kde(samples[:, dim], bw_method=0.2)
        x = np.linspace(min(samples[:, dim]), max(samples[:, dim]), 2000)
        density = kde(x)
        peaks, _ = find_peaks(density)

        return len(peaks), x, density

##################################################################################################

def calc_r_hat(chains):

    '''
    Calculates the Gelman-Rubin statistic for MCMC convergence
    '''

    n_chains,n_samples,n_params=chains.shape

    w=np.mean(np.var(chains,axis=1,ddof=1),axis=0)

    chain_means=np.mean(chains,axis=1)
    b=n_samples*np.var(chain_means,axis=0,ddof=1)

    var_plus=(1-1/n_samples)*w+(1/n_samples)*b

    r_hat=np.sqrt(var_plus/w)

    return r_hat

############################################################################

def calculate_pull(y_fit,y_known,e_fit):

    '''
    Calculates the pull a.k.a residual
    '''

    pull = (y_fit - y_known) / np.maximum(e_fit, 1e-12)
    
    return pull

#######################################################################################

def calculate_std_percent(y_fit,y_known,e_fit,std_coeff):

    '''
    Calculates the percentage of pulls that are within a standard deviation multiple
    '''

    pull=calculate_pull(y_fit,y_known,std_coeff*e_fit)
    
    percent=len([item for item in pull if abs(item)<=1])/len(pull)
    
    return percent

########################################################################################

def nll_gp(lengths, x_known, y_known, e_known):
    """
    Compute the negative log-likelihood (NLL) for your GP using your existing kernel_func.
    
    lengths: array of length scales (ndim,)
    x_known: (n_samples, n_features)
    y_known: (n_samples,)
    e_known: (n_samples,) observation noise
    """
    # Compute kernel matrix + noise
    K = kernel_func(x_known, x_known, lengths) + np.diag(e_known**2)
    
    # Ensure positive-definite
    try:
        L = cholesky(K)
    except np.linalg.LinAlgError:
        return 1e25  # return large number if K is not PD

    # Solve K^-1 y using Cholesky
    alpha = solve(L.T, solve(L, y_known))
    
    # Compute log determinant
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    
    # Negative log-likelihood
    nll = 0.5 * y_known.dot(alpha) + 0.5 * logdet + 0.5 * len(y_known) * np.log(2*np.pi)
    
    return nll

###########################################################################################

def find_nll(x_known,y_known,e_known):

    ndim = x_known.shape[1]
    initial_guess = np.ones(ndim) * 1  

    res = minimize(
        nll_gp, 
        initial_guess, 
        args=(x_known, y_known, e_known), 
        method='L-BFGS-B', 
        bounds=[(1e-5, None)]*ndim
    )

    optimal_lengths = res.x
    print("Optimal length scales:", optimal_lengths)

    return optimal_lengths,res.fun

#############################################################################################
