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
from scipy.optimize import minimize
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

def basic_pso(objective_func, bounds, init_positions=None, max_iter=200,
              w=0.7, c1=1.5, c2=2.0, verbose=False, random_seed=None):
    """
    Basic PSO that allows custom initial positions instead of a fixed particle count.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    ndim = len(bounds[0])
    lower, upper = np.array(bounds[0]), np.array(bounds[1])

    if init_positions is not None:
        pos = np.array(init_positions)
        num_particles = pos.shape[0]
    else:
        num_particles = 30
        pos = np.random.uniform(low=lower, high=upper, size=(num_particles, ndim))

    vel = np.zeros_like(pos)

    personal_best_pos = np.copy(pos)
    personal_best_val = np.array([objective_func(p) for p in pos])

    global_best_idx = np.argmin(personal_best_val)
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_val = personal_best_val[global_best_idx]

    loss_history = [global_best_val]

    for i in range(max_iter):
        r1 = np.random.rand(num_particles, ndim)
        r2 = np.random.rand(num_particles, ndim)
        vel = w * vel + c1 * r1 * (personal_best_pos - pos) + c2 * r2 * (global_best_pos - pos)
        pos += vel
        pos = np.clip(pos, lower, upper)

        scores = np.array([objective_func(p) for p in pos])

        better_mask = scores < personal_best_val
        personal_best_pos[better_mask] = pos[better_mask]
        personal_best_val[better_mask] = scores[better_mask]

        best_idx_now = np.argmin(personal_best_val)
        if personal_best_val[best_idx_now] < global_best_val:
            global_best_val = personal_best_val[best_idx_now]
            global_best_pos = personal_best_pos[best_idx_now].copy()

        loss_history.append(global_best_val)

        if verbose and i % 10 == 0:
            print(f"Iteration {i:3d} | Best loss: {-global_best_val:.6f}")

    return global_best_pos, global_best_val, loss_history



###############################################################################################

def len_scale_opt(x_known,y_known,e_known,MC_progress,MC_plotting,labels,out_file_name):  

    '''
    Finds the optimal length scale based on the given loss function. 
    '''

    original_file_path = Path(out_file_name)
    plotting_path = original_file_path.parent

    ndim=len(x_known)
    nwalkers=10*ndim
    max_n=2000*ndim

    r_hat_tol=1.16
    tau_tol=0.15

    diff = (np.max(x_known, axis=1) - np.min(x_known, axis=1)) / x_known.shape[1]
    endpoints = np.column_stack((np.full(x_known.shape[0], 1e-16), 10 * diff))

    
    sampling = LHS(xlimits=np.array(endpoints),criterion='center')
    initial_positions = sampling(nwalkers)
    perturb = np.random.normal(scale=0.2, size=initial_positions.shape)
    initial_positions = np.clip(initial_positions + perturb * initial_positions, 1e-16, None)

    
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
        sampler = emcee.EnsembleSampler(nwalkers,ndim,sigma_check,args=(x_known,y_known,e_known,sigma_vals,expected_percents),\
                                        backend=backend,pool=pool)
        for sample in sampler.sample(initial_positions,iterations=max_n,progress=MC_progress):
            if sampler.iteration%100:
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

    burnin=int(0.2*sampler.iteration)
    
    samples = sampler.get_chain(discard=burnin,thin=5,flat=True)

    #num_peaks,x,density=test_unimode(samples,dim=0)
    num_peaks, x, density, modes = test_unimode(samples, dim=0)

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
        return -sigma_check(lengths,x_known,y_known,e_known,sigma_vals,expected_percents)    
    
    #bounds=[(1e-16,None) for i in range(ndim)]

    score = 1e12
    best = None

    top_K = min(3, len(modes))  # Try refining top 3 modes
    for j in range(top_K):
        try:
            result = minimize(func_minimise, modes[j].flatten(), method="Nelder-Mead")
            if result.fun < score:
                best = result.x
                score = result.fun
        except Exception as e:
            print(f"Optimization failed for mode {j}: {e}")


    print("Optimal length scale found:")
    print(best)
    print("Value of loss function:")
    print(score)

    if os.path.exists(filename):
        os.remove(filename)

    return best



##############################################################################


def test_unimode(samples, dim=0, max_components=5):
    """
    Detects number of modes using Gaussian Mixture Model.
    """
    gmm = GaussianMixture(n_components=max_components, random_state=0)
    gmm.fit(samples)
    num_peaks = gmm.n_components
    x = np.linspace(np.min(samples[:, dim]), np.max(samples[:, dim]), 1000)
    kde = gaussian_kde(samples[:, dim])
    density = kde(x)

    return num_peaks, x, density, gmm.means_


##################################################################################################

def test_unimode1(samples,dim=0):

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


#########################################################################################

def single_len_scale_opt(x_known, y_known, e_known,
                        MC_progress=False,
                        MC_plotting=False,
                        labels=None,
                        out_file_name="output",
                        random_seed=None):
    
    ndim = x_known.shape[0]
    sigma_vals = np.linspace(0.001, 3, 1000)
    expected_percents = sigma_to_percent(sigma_vals)

    ranges = np.max(x_known, axis=1) - np.min(x_known, axis=1)
    lower_bounds = 0.01 * ranges
    upper_bounds = 2.0 * ranges

    def wrapped_loss(lengths):
        return -sigma_check(lengths, x_known, y_known, e_known, sigma_vals, expected_percents)

    if random_seed is not None:
        np.random.seed(random_seed)

    if MC_progress:
        print(f"[Seed {random_seed}] Starting PSO...")

    lhs = LHS(xlimits=np.array([lower_bounds, upper_bounds]).T)
    n_points = x_known.shape[1]  
    n_particles = max(10, int(np.ceil(1.1 * n_points)))  
    init_positions = lhs(n_particles)

    best_pos, best_val, loss_history = basic_pso(
        wrapped_loss,
        bounds=(lower_bounds, upper_bounds),
        init_positions=init_positions,
        max_iter=200,
        verbose=MC_progress
    )

    if MC_progress:
        print(f"[Seed {random_seed}] Refining with Nelder-Mead...")

    result = minimize(wrapped_loss, best_pos, method="Nelder-Mead")

    if MC_progress:
        print(f"[Seed {random_seed}] Best PSO:        {best_pos}")
        print(f"[Seed {random_seed}] Refined result:  {result.x}")
        print(f"[Seed {random_seed}] Final loss:     {-result.fun:.6f}")

    if MC_plotting:
        plt.figure(figsize=(8, 5))
        plt.plot(-np.array(loss_history), label=f"Seed {random_seed} PSO Best Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("PSO Convergence")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_file_name.replace(".h5", f"_pso_convergence_seed{random_seed}.png"))
        plt.show()

    return result.x

#################################################################################################

def optimise_seed(args):
    seed, x_known, y_known, e_known = args
    lengths = single_len_scale_opt(
        x_known, y_known, e_known,
        MC_progress=False,
        MC_plotting=False,
        random_seed=seed
    )
    sigma_vals = np.linspace(0.001, 3, 1000)
    expected_percents = sigma_to_percent(sigma_vals)
    score = sigma_check(lengths, x_known, y_known, e_known, sigma_vals, expected_percents)

    return seed, lengths, score

########################################################################################################

def parallel_len_scale_opt(x_known, y_known, e_known, n_runs=10, MC_progress=False):
    jobs = [(seed, x_known, y_known, e_known) for seed in range(n_runs)]
    results = []

    cpu_count = os.cpu_count() or 1
    max_workers = max(1, math.floor(cpu_count * 0.25))
    max_workers = min(n_runs, max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(optimise_seed, job): job[0] for job in jobs}

        if MC_progress:
            for future in tqdm(as_completed(futures), total=n_runs, desc="PSO seeds running"):
                res = future.result()
                results.append(res)
        else:
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda r: r[2], reverse=True)
    best_seed, best_lengths, best_score = results[0]
    if MC_progress:
        print(f"Best seed: {best_seed} with score: {best_score:.6f}")

    return best_lengths, results




