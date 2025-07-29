import numpy as np
from numpy.linalg import inv

from smt.sampling_methods import LHS
import emcee
import multiprocessing
import corner
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path
import os

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.stats import norm

from GP_func import GP

###################################################################################################

def sigma_check(lengths, x_known, y_known, e_known,sigma_vals,expected_percents):

    '''
    Computes the loss function:
    '''

    if np.any(lengths <= 0):
        return -1e13

    y_fit, e_fit = GP(x_known, y_known, e_known, x_known, lengths)
    
    scaled_e = e_fit[:, None] * sigma_vals[None, :] 

    pulls = (y_fit[:, None] - y_known[:, None]) / np.maximum(scaled_e, 1e-12)

    fractions_within = np.mean(np.abs(pulls) <= 1, axis=0)

    loss = -np.sum(np.abs(fractions_within - expected_percents))

    return loss

##########################################################################################

def sigma_to_percent(x):

    '''
    Calculates the percentage of points within x*standard deviation
    '''

    upper=norm.cdf(x)
    lower=norm.cdf(-x)

    return upper-lower

##################################################################################################################

def basic_pso(objective_func, bounds, num_particles=30, max_iter=200, w=0.7, c1=1.5, c2=2.0, verbose=False):

    ndim = len(bounds[0])
    lower, upper = np.array(bounds[0]), np.array(bounds[1])

    pos = np.random.uniform(low=lower, high=upper, size=(num_particles, ndim))
    vel = np.zeros_like(pos)

    personal_best_pos = np.copy(pos)
    personal_best_val = np.array([objective_func(p) for p in pos])

    global_best_idx = np.argmin(personal_best_val)
    global_best_pos = personal_best_pos[global_best_idx]
    global_best_val = personal_best_val[global_best_idx]

    loss_history = []

    for i in range(max_iter):
        r1, r2 = np.random.rand(num_particles, ndim), np.random.rand(num_particles, ndim)
        vel = w * vel + c1 * r1 * (personal_best_pos - pos) + c2 * r2 * (global_best_pos - pos)
        pos += vel
        pos = np.clip(pos, lower, upper)

        scores = np.array([objective_func(p) for p in pos])
        better_mask = scores < personal_best_val
        personal_best_pos[better_mask] = pos[better_mask]
        personal_best_val[better_mask] = scores[better_mask]

        global_best_idx = np.argmin(personal_best_val)
        if personal_best_val[global_best_idx] < global_best_val:
            global_best_pos = personal_best_pos[global_best_idx]
            global_best_val = personal_best_val[global_best_idx]

        loss_history.append(global_best_val)

        if verbose and i % 10 == 0:
            print(f"Iteration {i:3d} | Best loss: {-global_best_val:.6f}")

    return global_best_pos, global_best_val, loss_history


###############################################################################################

def len_scale_opt(x_known, y_known, e_known, MC_progress=False, MC_plotting=False, labels=None, out_file_name="output"):

    ndim = x_known.shape[0]

    sigma_vals = np.linspace(0.001, 3, 1000)
    expected_percents = sigma_to_percent(sigma_vals)

    diff = (np.max(x_known, axis=1) - np.min(x_known, axis=1)) / x_known.shape[1]
    lower_bounds = np.full(ndim, 1e-16)
    upper_bounds = 10 * diff

    def wrapped_loss(lengths):
        return -sigma_check(lengths, x_known, y_known, e_known, sigma_vals, expected_percents)

    if MC_progress:
        print("Starting PSO...")
    best_pos, best_val, loss_history = basic_pso(
        wrapped_loss,
        bounds=(lower_bounds, upper_bounds),
        num_particles=40,
        max_iter=200,
        verbose=MC_progress
    )

    if MC_progress:
        print("\nRefining best result with Nelder-Mead...")
    result = minimize(wrapped_loss, best_pos, method="Nelder-Mead")

    if MC_progress:
        print(f"\nBest from PSO:           {best_pos}")
        print(f"Refined length scales:   {result.x}")
        print(f"Final loss:             {-result.fun:.6f}")

    if MC_plotting:
        plt.figure(figsize=(8,5))
        plt.plot(-np.array(loss_history), label="Best loss (PSO)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss (sigma_check)")
        plt.title("PSO Convergence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_file_name.replace(".h5", "_pso_convergence.png"))
        plt.show()

    return result.x


##############################################################################

def test_unimode(samples,dim=0):

    '''
    Tests if the length scale surface is unimodal
    '''    

    kde = gaussian_kde(samples[:,dim])
    x=np.linspace(min(samples[:,dim]),max(samples[:,dim]),1000)
    density=kde(x)

    peaks,temp=find_peaks(density)

    return len(peaks),x,density

###################################################################################

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