import os
import numpy as np
from numpy.linalg import inv
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

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.optimize import minimize,differential_evolution
from scipy.stats import norm

from sklearn.mixture import GaussianMixture

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

def len_scale_opt(x_known, y_known, e_known, MC_progress=False, MC_plotting=False,
                  labels=None, out_file_name="output.txt",
                  num_particles=30, max_iters=200, patience=30, refine=True):
    '''
    Enhanced single-run PSO for optimal length scale using:
    - Dynamic inertia
    - Velocity clamping
    - Latin Hypercube Sampling
    - Early stopping
    - Optional local refinement
    '''

    ndim = len(x_known)
    diff = np.max(x_known, axis=1) - np.min(x_known, axis=1)
    lower_bounds = 0.01 * diff
    upper_bounds = 2.0 * diff
    bounds = [(l, u) for l, u in zip(lower_bounds, upper_bounds)]

    sigma_vals = np.linspace(0.001, 3, 1000)
    expected_percents = sigma_to_percent(sigma_vals)

    def func_minimise(lengths):
        return -sigma_check(lengths, x_known, y_known, e_known, sigma_vals, expected_percents)

    def lhs(n, samples):
        seg = np.linspace(0, 1, samples + 1)
        temp = np.random.rand(samples, n) / samples
        points = seg[:samples, None] + temp
        np.random.shuffle(points)
        return points

    lhs_samples = lhs(ndim, num_particles)
    positions = lower_bounds + lhs_samples * (upper_bounds - lower_bounds)
    velocities = np.zeros_like(positions)

    personal_best_positions = positions.copy()
    personal_best_scores = np.array([func_minimise(p) for p in positions])
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_score = personal_best_scores[global_best_index]

    c1, c2 = 1.5, 1.5
    v_max = 0.2 * (upper_bounds - lower_bounds)
    best_score_history = [global_best_score]
    no_improve_counter = 0

    for i in range(max_iters):
        w = 0.9 - (0.5 * i / max_iters)

        r1 = np.random.rand(num_particles, ndim)
        r2 = np.random.rand(num_particles, ndim)

        velocities = (w * velocities +
                      c1 * r1 * (personal_best_positions - positions) +
                      c2 * r2 * (global_best_position - positions))
        velocities = np.clip(velocities, -v_max, v_max)

        positions += velocities
        positions = np.clip(positions, lower_bounds, upper_bounds)

        for j in range(num_particles):
            score = func_minimise(positions[j])
            if score < personal_best_scores[j]:
                personal_best_scores[j] = score
                personal_best_positions[j] = positions[j].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[j].copy()
                    no_improve_counter = 0
        else:
            no_improve_counter += 1

        best_score_history.append(global_best_score)

        if MC_progress and (i % 20 == 0 or i == max_iters - 1):
            print(f"Iter {i:4d} | Best Score: {global_best_score:.6f}")

        if no_improve_counter >= patience:
            if MC_progress:
                print(f"Stopping early at iteration {i} — no improvement for {patience} steps.")
            break

    if refine:
        if MC_progress:
            print("Refining with L-BFGS-B...")
        result = minimize(func_minimise, global_best_position, method='L-BFGS-B', bounds=bounds)
        global_best_position = result.x
        global_best_score = result.fun

    print("\nFinal optimal length scale:")
    print(global_best_position)
    print("Final loss:")
    print(global_best_score)

    return global_best_position


##############################################################################

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

