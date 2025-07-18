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

##############################################################################################

def len_scale_opt(x_known,y_known,e_known,MC_progress,MC_plotting,labels,out_file_name):  

    '''
    Finds the optimal length scale based on the given loss function. 
    '''

    original_file_path = Path(out_file_name)
    plotting_path = original_file_path.parent

    ndim=len(x_known)
    nwalkers=8*ndim
    max_n=2000*ndim

    r_hat_tol=1.18
    tau_tol=0.15

    diff = (np.max(x_known, axis=1) - np.min(x_known, axis=1)) / x_known.shape[1]
    endpoints = np.column_stack((np.full(x_known.shape[0], 1e-16), 10 * diff))

    
    sampling = LHS(xlimits=np.array(endpoints),criterion='center')
    initial_positions = sampling(nwalkers)
    
    filename="backend.h5"
    backend=emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers,ndim)

    print("Beginning MCMC")

    sigma_vals = np.linspace(0.001, 3, 1000) 
    expected_percents = sigma_to_percent(sigma_vals)

    with multiprocessing.Pool(16) as pool:

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
            
            tau_conv=np.all((np.abs(old_tau-tau)/tau)<tau_tol)
            if MC_progress:
                print("tau_stability:\t"+str(np.abs(old_tau-tau)/tau))

            chains=sampler.get_chain(discard=50,thin=5,flat=False)
            if chains.shape[1]>1:
                r_hat=calc_r_hat(chains)
                r_hat_conv=np.all(r_hat<r_hat_tol)
                if MC_progress:
                    print("r_hat:\t\t"+str(r_hat))
            
            if r_hat_conv and tau_conv:
                break
            old_tau=tau

    burnin=int(0.2*sampler.iteration)
    
    samples = sampler.get_chain(discard=burnin,thin=5,flat=True)

    num_peaks,x,density=test_unimode(samples,dim=0)

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

    return best

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