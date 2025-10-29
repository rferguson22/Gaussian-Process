import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from smt.sampling_methods import LHS
import multiprocessing
from sklearn.cluster import KMeans
from pathlib import Path

from scipy.stats import ks_2samp
from scipy.stats import norm

from GP_func import GP

###################################################################################################

def len_scale_opt(x_known, y_known, e_known, PSO_progress):
    

    max_points = 100
    if x_known.shape[1] > max_points:
        print(f"Dataset too large ({x_known.shape[1]} points). Subsampling to {max_points} for hyperparameter optimisation.")

        # Fit KMeans on the data (transpose: (dims, points) → (points, dims))
        X = x_known.T
        kmeans = KMeans(n_clusters=max_points, n_init='auto', random_state=0)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        # Pick closest actual point to each cluster center
        idx = []
        for k in range(max_points):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                continue  # skip empty clusters (rare)
            d2 = np.sum((X[members] - centers[k])**2, axis=1)
            closest = members[np.argmin(d2)]
            idx.append(closest)

        idx = np.array(idx)

        # Apply the chosen indices
        x_known = x_known[:, idx]
        y_known = y_known[idx]
        e_known = e_known[idx]
    
    ndim = len(x_known)
    num_particles = 40
    max_iter = 500
    patience = 100
    inertia_decay = 0.002
    restart_count = 0

    ranges = np.max(x_known, axis=1) - np.min(x_known, axis=1)
    lower_bounds = 0.01 * ranges / x_known.shape[1]  
    upper_bounds = ranges
    bounds_array = np.column_stack((lower_bounds, upper_bounds))
    v_max = 1.0 * (upper_bounds - lower_bounds) 

    sigma_vals = np.linspace(0.001, 3, 1000)
    expected_percents = sigma_to_percent(sigma_vals)

    sampling = LHS(xlimits=bounds_array, criterion="center")
    positions = sampling(num_particles)
    velocities = np.zeros_like(positions)

    num_cores = max(1, os.cpu_count() // 4)
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=num_cores) as executor:
        # Evaluate initial personal bests once
        args_iterable = [(p, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds,upper_bounds) for p in positions]
        personal_best_scores = list(executor.map(evaluate_loss_helper, args_iterable))
        personal_best_scores = np.array(personal_best_scores)
        personal_best_positions = positions.copy()

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        no_improve_counter = 0

        for i in range(max_iter):
            w = max(0.4, 0.9 - i * inertia_decay)
            c1 = c2 = 1.4

            r1 = np.random.rand(num_particles, ndim)
            r2 = np.random.rand(num_particles, ndim)

            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))
            velocities = np.clip(velocities, -v_max, v_max)

            positions += velocities
            positions = np.clip(positions, lower_bounds, upper_bounds)

            args_iterable = [(p, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds,upper_bounds) for p in positions]
            scores = list(executor.map(evaluate_loss_helper, args_iterable))
            scores = np.array(scores)

            improved = scores < personal_best_scores
            personal_best_positions[improved] = positions[improved]
            personal_best_scores[improved] = scores[improved]

            current_best_idx = np.argmin(personal_best_scores)
            current_best_score = personal_best_scores[current_best_idx]

            if current_best_score < global_best_score:
                global_best_score = current_best_score
                global_best_position = personal_best_positions[current_best_idx].copy()
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if PSO_progress and i % 20 == 0:
                print(f"Iter {i}: Best Score = {global_best_score:.6f}, No Improve = {no_improve_counter}")

            if no_improve_counter >= patience:
                if PSO_progress:
                    print(f"Stagnation at iter {i}, soft-restarting swarm...")

                noise = 0.1 * (upper_bounds - lower_bounds)
                personal_best_positions += np.random.uniform(-noise, noise, personal_best_positions.shape)
                personal_best_positions = np.clip(personal_best_positions, lower_bounds, upper_bounds)
                positions = personal_best_positions.copy()
                velocities = np.zeros_like(positions)

                args_iterable = [(p, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds,upper_bounds) for p in personal_best_positions]
                personal_best_scores = list(executor.map(evaluate_loss_helper, args_iterable))
                personal_best_scores = np.array(personal_best_scores)

                global_best_idx = np.argmin(personal_best_scores)
                global_best_position = personal_best_positions[global_best_idx].copy()
                global_best_score = personal_best_scores[global_best_idx]
                no_improve_counter = 0
                restart_count += 1

    print("PSO completed.")
    print("Optimal length scale found:")
    print(global_best_position)
    print("Value of loss function:")
    print(global_best_score)
    print(f"Total soft restarts: {restart_count}")

    return global_best_position,global_best_score

    ###############################################################################################

########################################################################################################

def ks_loss(ls, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds, upper_bounds):

    if np.any(ls <= lower_bounds) or np.any(ls >= upper_bounds):
        return -1e13
    
    y_fit, e_fit = GP(x_known, y_known, e_known, x_known, ls, batch_size=x_known.shape[1])
    
    scaled_e = e_fit[:, None] * sigma_vals[None, :]
    pulls = (y_fit[:, None] - y_known[:, None]) / np.maximum(scaled_e, 1e-12)
    measured_percents = np.mean(np.abs(pulls) <= 1, axis=0)
    
    D, _ = ks_2samp(expected_percents, measured_percents)
    sigma_loss = np.mean(np.abs(measured_percents - expected_percents))
    
    return -D - (0.01 * sigma_loss)

#####################################################################################################

def evaluate_loss(lengths, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds,upper_bounds):
    return -ks_loss(lengths, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds,upper_bounds)

#################################################################################################################

def evaluate_loss_helper(args):
    p, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds,upper_bounds = args
    return evaluate_loss(p, x_known, y_known, e_known, sigma_vals, expected_percents, lower_bounds,upper_bounds)

#################################################################################################

def sigma_to_percent(x):

    '''
    Calculates the percentage of points within x*standard deviation
    '''

    upper=norm.cdf(x)
    lower=norm.cdf(-x)

    return upper-lower