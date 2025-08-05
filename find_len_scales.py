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

def sigma_check(lengths, x_known, y_known, e_known,sigma_vals,expected_percents,lower_bounds):

    '''
    Computes the loss function:
    '''

    if np.any(lengths <= lower_bounds):
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

def len_scale_swarm(x_known, y_known, e_known, MC_progress=False, MC_plotting=False,
                  labels=None, out_file_name="output.txt",
                  num_particles=30, max_iters=300, patience=40, refine=True):
    
    '''
    Enhanced PSO with:
    - Dynamic inertia (exponential decay)
    - Larger velocity bounds
    - Noise injection to personal bests
    - Soft restart if stagnated
    - Optional local refinement (L-BFGS-B)
    '''

    ndim = len(x_known)
    num_particles = 30
    max_iter = 500
    patience = 100
    inertia_decay = 0.002
    restart_count = 0

    diff = (np.max(x_known, axis=1) - np.min(x_known, axis=1))
    lower_bounds = 0.01 * diff
    upper_bounds = 2.0 * diff
    bounds_array = np.column_stack((lower_bounds, upper_bounds))
    v_max = 1.0 * (upper_bounds - lower_bounds)

    sigma_vals = np.linspace(0.001, 3, 1000)
    expected_percents = sigma_to_percent(sigma_vals)

    def loss_func(lengths):
        return -sigma_check(lengths, x_known, y_known, e_known, sigma_vals, expected_percents,lower_bounds)

    sampling = LHS(xlimits=bounds_array, criterion="center")
    positions = sampling(num_particles)
    velocities = np.zeros_like(positions)

    personal_best_positions = positions.copy()
    personal_best_scores = np.array([loss_func(p) for p in positions])

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

        scores = np.array([loss_func(p) for p in positions])

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

        if MC_progress and i % 20 == 0:
            print(f"Iter {i}: Best Score = {global_best_score:.6f}, No Improve = {no_improve_counter}")

        # Soft restart if no improvement
        if no_improve_counter >= patience:
            if MC_progress:
                print(f"Stagnation at iter {i}, soft-restarting swarm...")
            noise = 0.1 * (upper_bounds - lower_bounds)
            personal_best_positions += np.random.uniform(-noise, noise, personal_best_positions.shape)
            personal_best_positions = np.clip(personal_best_positions, lower_bounds, upper_bounds)
            positions = personal_best_positions.copy()
            velocities = np.zeros_like(positions)
            personal_best_scores = np.array([loss_func(p) for p in personal_best_positions])
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

##############################################################################

def len_scale_mcmc(x_known,y_known,e_known,MC_progress,MC_plotting,labels,out_file_name):  

    '''
    Finds the optimal length scale based on the given loss function. 
    '''

    original_file_path = Path(out_file_name)
    plotting_path = original_file_path.parent

    ndim=len(x_known)
    nwalkers=20*ndim
    max_n=2000*ndim

    r_hat_tol=1.1
    tau_tol=0.2

    diff = (np.max(x_known, axis=1) - np.min(x_known, axis=1))
    lower_bounds = 0.01 * diff
    upper_bounds = 2.0 * diff
    endpoints = np.column_stack((lower_bounds, upper_bounds))

    
    sampling = LHS(xlimits=np.array(endpoints),criterion='center')
    initial_positions = sampling(nwalkers)
    initial_positions += 1e-5 * np.random.randn(*initial_positions.shape)
    
    filename="backend.h5"
    backend=emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers,ndim)

    print("Beginning MCMC")

    sigma_vals = np.linspace(0.001, 3, 1000) 
    expected_percents = sigma_to_percent(sigma_vals)

    num_cores = max(1, os.cpu_count() // 4)
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=num_cores) as pool:
        index=0
        autocorr=np.empty(max_n)
        r_hat_conv=False
        
        old_tau=np.inf
        sampler = emcee.EnsembleSampler(nwalkers,ndim,sigma_check,args=(x_known,y_known,e_known,sigma_vals,expected_percents,lower_bounds),\
                                        backend=backend,pool=pool)
        for sample in sampler.sample(initial_positions,iterations=max_n,progress=MC_progress):
            if sampler.iteration % 100 != 0:
                continue
        
            tau=sampler.get_autocorr_time(tol=0)
            autocorr[index]=np.mean(tau)
            index+=1

            chains=sampler.get_chain(discard=50,thin=5,flat=False)
            if chains.shape[1]>1:
                r_hat=calc_r_hat(chains)
                r_hat_conv=np.all(r_hat<r_hat_tol)
                if MC_progress:
                    print("r_hat:\t\t"+str(r_hat))
            
            tau_conv=np.all((np.abs(old_tau-tau)/tau)<tau_tol)
            if MC_progress:
                print("tau_stability:\t"+str(np.abs(old_tau-tau)/tau))
            
            if r_hat_conv and tau_conv:
                break
            old_tau=tau

    burnin = int(0.5 * sampler.iteration)
    samples = sampler.get_chain(discard=burnin, thin=10, flat=True)

    num_peaks, x, density = test_unimode(samples, dim=0)

    print("MCMC converged. Checking for multimodal surface")

    if MC_plotting:
    
        fig=corner.corner(samples,labels=labels[:-2])
        fig.savefig(plotting_path+"/GP_corner_plot.png")
    
        plt.figure(figsize=(8,6))
        plt.plot(x,density,label="KDE")
        plt.title(f"KDE and Peak Detection (Peaks Found: {num_peaks})")
        plt.xlabel("Dim 1")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(plotting_path+"/GP_KDE_plots.png")
        plt.show()

    if num_peaks>1:
        print("Multimodal distribution detected. Performing Clustering")

        silhouette_scores=[]
        K_values=range(2,11)

        for K in K_values:
            kmeans=KMeans(n_clusters=K)
            cluster_labels=kmeans.fit_predict(samples)
            score=silhouette_score(samples,cluster_labels)
            silhouette_scores.append(score)

        if MC_plotting:
            plt.figure(figsize=(8,6))
            plt.plot(K_values,silhouette_scores,"-o",color="blue")
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Score for Optimal K")
            plt.grid(True)
            plt.savefig(plotting_path+"/GP_silhouette_scores.png")
            plt.show()

        optimal_K=K_values[np.argmax(silhouette_scores)]
        print(f"Optimal number of Clusters (K): {optimal_K}")

        kmeans=KMeans(n_clusters=optimal_K)
        cluster_labels=kmeans.fit_predict(samples)
        modes=kmeans.cluster_centers_

    else:
        print("Single mode surface")
        modes=np.mean(samples,axis=0,keepdims=True)

    print("Modes of surface:")
    print(modes)
    
    def func_minimise(lengths):
        return -sigma_check(lengths,x_known,y_known,e_known,sigma_vals,expected_percents,lower_bounds)    
    
    bounds = [(1e-16, ub) for ub in upper_bounds]
    score = np.inf
    best = None

    for j in range(len(modes)):
        result = differential_evolution(func_minimise, bounds=bounds, strategy='best1bin', maxiter=1000)
        if result.fun < score:
            best = result.x
            score = result.fun


    print("Optimal length scale found:")
    print(best)
    print("Value of loss function:")
    print(score)

    if os.path.exists(filename):
        os.remove(filename)

    return best,score

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

