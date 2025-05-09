import numpy as np
from numpy import random
from numpy.linalg import inv

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit,minimize
from scipy.special import legendre
from scipy.stats.distributions import norm
from scipy.stats import poisson,gaussian_kde
import scipy

from iminuit import Minuit
from iminuit.cost import LeastSquares

import matplotlib.pyplot as plt

import emcee
import corner
import math
import pandas as pd 
from smt.sampling_methods import LHS
import multiprocessing

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
import corner
from collections import Counter

#############################################################################################################

def kernel_func(xy1, xy2,l):
    
    sq_norm = -0.5*((cdist(xy1[:1].T, xy2[:1].T, 'sqeuclidean')/((l[0])**2))+
    (cdist(xy1[1:].T, xy2[1:].T, 'sqeuclidean')/((l[1])**2)))
                      
    rbf=np.exp(sq_norm)  
    
    if len(xy1)!=len(l):
        rbf*=l[-1]
    
    return rbf

##############################################################################################################

def GP(xy_known, z_known, e_known,xy,lengths):

    K = kernel_func(xy_known, xy_known,lengths) + e_known**2 * np.eye(len(e_known))
    K_s = kernel_func(xy_known,xy,lengths)
    K_ss = kernel_func(xy,xy,lengths) 
    K_inv = inv(K)
        
    mu_s = K_s.T.dot(K_inv).dot(z_known)

    sigma_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, np.sqrt(abs(np.diag(sigma_s)))

############################################################################################################

def create_convex_hull_boundary(xy_known,accuracy_e,accuracy_a):
    
    known_e,temp=[],[]
    decimal_places_e=str(accuracy_e)[::-1].find('.')
    decimal_places_a=str(accuracy_a)[::-1].find('.')
    
    i=0
    while i <len(xy_known.T)-1:
        temp.append(xy_known.T[i])
        if xy_known.T[i,0]!=xy_known.T[i+1,0]:
            known_e.append(np.sort(temp))
            temp=[]    
        i+=1

    temp.append(xy_known.T[-1])
    known_e.append(np.sort(temp))

    l_bound,r_bound=[],[]
    j=0
    while j <(len(known_e)):
        l_temp,r_temp=[],[]
        l_temp.append(known_e[j][0][1])
        l_temp.append(known_e[j][0][0])

        r_temp.append(known_e[j][-1][1])
        r_temp.append(known_e[j][-1][0])

        if j==len(known_e)-1:
            l_temp.append(0.5*abs(known_e[j][0][1]-known_e[j-1][0][1]))
            r_temp.append(0.5*abs(known_e[j][-1][1]-known_e[j-1][-1][1]))
        elif j==0:
            l_temp.append(0.5*abs(known_e[j][0][1]-known_e[j+1][0][1]))
            r_temp.append(0.5*abs(known_e[j][-1][1]-known_e[j+1][-1][1]))
        else:
            l_temp.append(0.25*(abs(known_e[j][0][1]-known_e[j+1][0][1])+\
                         abs(known_e[j][0][1]-known_e[j-1][0][1])))
            r_temp.append(0.25*(abs(known_e[j][-1][1]-known_e[j+1][-1][1])+\
                         abs(known_e[j][-1][1]-known_e[j-1][-1][1])))

        if len(known_e)>1:
            l_temp.append(0.5*abs(known_e[j][0][0]-known_e[j][1][0]))
            r_temp.append(0.5*abs(known_e[j][-1][0]-known_e[j][-2][0]))
        else:
            l_temp.append(0)
            r_temp.append(0)

        l_bound.append(l_temp)
        r_bound.append(r_temp)

        j+=1

    l_bound=np.array(l_bound)
    r_bound=np.array(r_bound)
    l_bound.T[3]=[np.mean([item for item in l_bound.T[3] if item!=0]) \
                  if item==0 else item for item in l_bound.T[3]]
    r_bound.T[3]=[np.mean([item for item in r_bound.T[3] if item!=0]) \
                  if item==0 else item for item in r_bound.T[3]]

    points=[]
    for i in range(len(l_bound)):
        points.append([round(l_bound[i][0]-l_bound[i][2],decimal_places_e),\
                       round(l_bound[i][1]-l_bound[i][3],decimal_places_a)])
        points.append([round(r_bound[i][0]-r_bound[i][2],decimal_places_e),\
                       round(r_bound[i][1]+r_bound[i][3],decimal_places_a)])
        points.append([round(l_bound[i][0]+l_bound[i][2],decimal_places_e),\
                       round(l_bound[i][1]-l_bound[i][3],decimal_places_a)])
        points.append([round(r_bound[i][0]+r_bound[i][2],decimal_places_e),\
                       round(r_bound[i][1]+r_bound[i][3],decimal_places_a)])
        
    points=np.array(points)
    
    points.T[1]=[1 if item>1 else item for item in points.T[1]]
    points.T[1]=[-1 if item<-1 else item for item in points.T[1]]
    
    xy=create_convex_hull(points,accuracy_e,accuracy_a)
    
    return xy    

#####################################################################################################

def create_convex_hull(points,accuracy_e,accuracy_a):
    
    decimal_places_e=str(accuracy_e)[::-1].find('.')
    decimal_places_a=str(accuracy_a)[::-1].find('.')

    hull=ConvexHull(points)

    corners=points[hull.simplices]

    hull_boundary_e=[]
    hull_boundary_a=[]

    for i in range(len(corners)):

        if corners[i][0][0]<corners[i][1][0]:
            start=corners[i][0][0]
            end=corners[i][1][0]
        else:
            start=corners[i][1][0]
            end=corners[i][0][0]

        if hull.equations[i][1]!=0:

            if end==max(points.T[0]):
                addition=accuracy_e/2
            else:
                addition=0


            energies=np.around(np.arange(start,end+addition,accuracy_e),decimal_places_e)
            energies=np.array([item for item in energies if (item!=end) or\
                      (item==end and addition!=0)])
            angles=np.around(-(hull.equations[i][0]*energies+hull.equations[i][2])\
                             /hull.equations[i][1],2)
            hull_boundary_e=np.concatenate((hull_boundary_e,energies))
            hull_boundary_a=np.concatenate((hull_boundary_a,angles))

    hull_boundary=np.column_stack((hull_boundary_e,hull_boundary_a)).T

    e=np.around(np.arange(min(hull_boundary[0]),max(hull_boundary[0])+accuracy_e/2,accuracy_e),decimal_places_e)

    h_bound=[]
    xy=[]
    
    for i in range(len(e)):
        temp=np.argwhere(hull_boundary[0]==e[i])
        if hull_boundary[1][temp[0][0]]<hull_boundary[1][temp[1][0]]:
            lower=hull_boundary[1][temp[0][0]]
            upper=hull_boundary[1][temp[1][0]]
        else:
            lower=hull_boundary[1][temp[1][0]]
            upper=hull_boundary[1][temp[0][0]]        

        angle=lower
        while angle<=upper:
            xy.append([e[i],angle])
            angle+=accuracy_a
            angle=round(angle,decimal_places_a)


    xy=np.array(xy)
    
    return xy.T

##############################################################################################

def assign_known_parameters(energy_limits,measured_angle_limits,\
                            datapoints_energies_measured,datapoints_angles_measured):

    x_known,y_known=[],[]

    energies_range=abs(energy_limits[0]-energy_limits[1])/datapoints_energies_measured 
    
    for i in range(datapoints_energies_measured):

        energy_temp=math.floor(random.uniform((energy_limits[0]+(i*energies_range)),\
                                         (energy_limits[0]+((i+1)*energies_range)))*100)/100

        
        known_points_no=int(random.uniform(datapoints_angles_measured[0],\
                                           datapoints_angles_measured[1]))
            
        
        angles_range=abs(measured_angle_limits[0]-measured_angle_limits[1])/known_points_no

        for j in range(known_points_no):

            angle_temp=math.floor(random.uniform((measured_angle_limits[0]+(j*angles_range)),\
                                            (measured_angle_limits[0]+((j+1)*angles_range)))*100)/100
            
            x_known.append(energy_temp)
            y_known.append(angle_temp)

    
    xy_known=np.column_stack((x_known,y_known)).T
        
    return xy_known

##############################################################################################################

def generate_pseudo_func(xy,coeff_all,legendre_orders):

    z_func=np.zeros(len(xy.T))

    for i in range(len(z_func)):
        j=0
        while j<len(coeff_all):
            order=legendre_orders[int(j/3)]
            b=(norm.pdf(xy[0][i],coeff_all[j+1],coeff_all[j+2])\
                        *coeff_all[j]*legendre(order)(xy[1][i]))
            z_func[i]+=b
            j+=3

    x=np.linspace(0.6,0.9,1000)
    a=x*(1-x)
    p=a/np.sum(a)
    scale=np.random.choice(x,p=p)/max(abs(z_func))
    z_func*=scale
    k=0
    while k<len(coeff_all):
        coeff_all[k]*=scale
        k+=3

    return z_func,coeff_all

#################################################################################################################

def generate_pseudo_data(xy_known,xy,z_func,eff_count_limits):

    z_known,e_known,z_func_known=[],[],[]   
    

    for i in range(len(xy_known.T)):
        
        e_temp=np.argwhere(xy_known.T[i][0]==xy[0])
        a_temp=np.argwhere(xy_known.T[i][1]==xy[1])
        coeff_temp=[item for item in a_temp if item in e_temp][0][0]
        
        a=z_func[coeff_temp]

        effective_count= np.random.uniform(low=eff_count_limits[0], high=eff_count_limits[1])

        n_plus=0.5*effective_count*(1+a)
        n_minus=0.5*effective_count*(1-a)

        N_plus=poisson.rvs(n_plus)
        N_minus=poisson.rvs(n_minus)

        z=(N_plus-N_minus)/(N_plus+N_minus)
        e=(2/((N_plus+N_minus)**2))*np.sqrt(N_plus*N_minus*(N_plus+N_minus))


        z_known.append(z)
        e_known.append(e)
        z_func_known.append(a)
            

    z_known=np.array(z_known)
    e_known=np.array(e_known) 
    z_func_known=np.array(z_func_known)
    
        
    return z_known,e_known,z_func_known

######################################################################################################

def calculate_std_percent(z_fit,z_func,e_fit,std_coeff):
    
    
    pull=calculate_pull(z_fit,z_func,std_coeff*e_fit)
    
    percent=len([item for item in pull if abs(item)<=1])/len(pull)
    
    return percent

########################################################################################################

def len_scale_opt(x_known_temp,y_known_temp,e_known_temp,func):  
    
    ndim=len(x_known_temp)
    nwalkers=8*ndim
    max_n=2000*ndim
    
    endpoints=[]
    for i in range(len(x_known_temp)):
        diff=(max(x_known_temp[i])-min(x_known_temp[i]))/len(x_known_temp[i].T)
        endpoints.append([1e-16,10*diff])
    
    sampling = LHS(xlimits=np.array(endpoints),criterion='center')
    initial_positions = sampling(nwalkers)
    
    filename="backend.h5"
    backend=emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers,ndim)
        
    with multiprocessing.Pool(16) as pool:

        index=0
        autocorr=np.empty(max_n)
        r_hat_conv=False
        
        old_tau=np.inf
        sampler = emcee.EnsembleSampler(nwalkers,ndim,func,args=(x_known_temp,y_known_temp,e_known_temp),backend=backend,pool=pool)
        for sample in sampler.sample(initial_positions,iterations=max_n,progress=True):
            if sampler.iteration%100:
                continue
        
            tau=sampler.get_autocorr_time(tol=0)
            autocorr[index]=np.mean(tau)
            index+=1
            
            tau_conv=np.all((np.abs(old_tau-tau)/tau)<tau_tol)
            print(np.abs(old_tau-tau)/tau)

            chains=sampler.get_chain(discard=50,thin=5,flat=False)
            if chains.shape[1]>1:
                r_hat=calc_r_hat(chains)
                print(r_hat)
                r_hat_conv=np.all(r_hat<r_hat_tol)
            
            if r_hat_conv and tau_conv:
                break
            old_tau=tau

    burnin=int(0.2*sampler.iteration)
    
    samples = sampler.get_chain(discard=burnin,thin=5,flat=True)

    labels=["L1","L2"]

    fig=corner.corner(samples,labels=labels)
    fig.savefig("corner_plot.png")

    num_peaks,x,density=test_unimode(samples,dim=0)

    plt.figure(figsize=(8,6))
    plt.plot(x,density,label="KDE")
    plt.title(f"KDE and Peak Detection (Peaks Found: {num_peaks})")
    plt.xlabel("Dim 1")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    plt.close()

    if num_peaks>1:
        print("Multimodal distribution detected. Performing Clustering")

        silhouette_scores=[]
        K_values=range(2,11)

        for K in K_values:
            kmeans=KMeans(n_clusters=K)
            cluster_labels=kmeans.fit_predict(samples)
            score=silhouette_score(samples,cluster_labels)
            silhouette_scores.append(score)
            
        plt.figure(figsize=(8,6))
        plt.plot(K_values,silhouette_scores,"-o",color="blue")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score for Optimal K")
        plt.grid(True)
        plt.show()
        plt.close()

        optimal_K=K_values[np.argmax(silhouette_scores)]
        print(f"Optimal number of Clusters (K): {optimal_K}")

        kmeans=KMeans(n_clusters=optimal_K)
        cluster_labels=kmeans.fit_predict(samples)
        modes=kmeans.cluster_centers_

    else:
        modes=np.mean(samples,axis=0,keepdims=True)


    print(modes)
    
    def func_minimise(lengths):
        return -func(lengths,x_known_temp,y_known_temp,e_known_temp)    
    
    bounds=[(1e-16,None) for i in range(ndim)]

    score=1e12
    best=[]
    
    for j in range(len(modes)):
        result=minimize(func_minimise,modes[j],method="Nelder-Mead",bounds=bounds)
        if result.fun<score:
            best=result.x
            score=result.fun

    print(best)
    print(score)

    #return best,score
    return best

####################################################################################

def sigma_to_percent(x):
    upper=scipy.stats.norm.cdf(x)
    lower=scipy.stats.norm.cdf(-x)

    return upper-lower

#####################################################################################

def sigma_check(lengths,x_known_temp,y_known_temp,e_known_temp):

    if min(lengths)<0:
        return -10e12

    y_fit,e_fit=GP(x_known_temp,y_known_temp,e_known_temp,x_known_temp,lengths)
    
    sigma=np.linspace(0.001,3,1000)
    total=0

    for i in range(len(sigma)):
        percent = sigma_to_percent(sigma[i])
        total-=abs(calculate_std_percent(y_fit,y_known_temp,e_fit,sigma[i])-percent)
    
    return total

############################################################################################

def test_unimode(samples,dim=0):
    
    kde = gaussian_kde(samples[:,dim])
    x=np.linspace(min(samples[:,dim]),max(samples[:,dim]),1000)
    density=kde(x)

    peaks,temp=find_peaks(density)

    return len(peaks),x,density

############################################################################################

def calc_r_hat(chains):

    n_chains,n_samples,n_params=chains.shape

    w=np.mean(np.var(chains,axis=1,ddof=1),axis=0)

    chain_means=np.mean(chains,axis=1)
    b=n_samples*np.var(chain_means,axis=0,ddof=1)

    var_plus=(1-1/n_samples)*w+(1/n_samples)*b

    r_hat=np.sqrt(var_plus/w)

    return r_hat

##########################################################################################

def calculate_pull(z_fit,z_func,e_fit):

    pull=np.zeros(len(z_fit))
    
    e_fit[e_fit==0]=1e-12

    for i in range(len(z_func)):
        pull[i]=(z_fit[i]-z_func[i])/(e_fit[i])

    return np.array(pull)

############################################################################################

def fit_data(xy_points,z_points,e_points,coeff_limits,gaus_mean_limits,gaus_width_limits,coeff_all):
    
    initial_form=[np.mean(coeff_limits),np.mean(gaus_mean_limits),np.mean(gaus_width_limits)]
    initial=np.array(initial_form*(int(len(coeff_all)/3)))
    
    mi=Minuit(LeastSquares(xy_points,z_points,e_points,model),initial)
    
    i=0
    
    while i<(int(len(coeff_all))):
        mi.limits["x"+str(i)]=coeff_limits
        mi.limits["x"+str(i+1)]=gaus_mean_limits
        mi.limits["x"+str(i+2)]=gaus_width_limits
        i+=3
        
    mi.migrad()
    mi.hesse()
    
    coeff_guess=np.array(mi.values)
    coeff_guess_error=np.array(mi.errors)
    
    pull=calculate_pull(coeff_guess,coeff_all,coeff_guess_error)
        
    return pull,np.array(mi.values)

##############################################################################################################

def model(xy_points,theta):

    i=0
    value=0

    while i<len(theta):
        order=legendre_orders[int(i/3)]
        value+=(theta[i]*norm.pdf(xy_points[0],theta[i+1],theta[i+2])*\
                legendre(order)(xy_points[1]))
        i+=3

    return value

#############################################################################################################

def get_xy_rand(xy,xy_known,accuracy_a,accuracy_e):
    xlimits=[[min(xy[0]),max(xy[0])],[min(xy[1]),max(xy[1])]]
    mul_fac=1
    sampling = LHS(xlimits=np.array(xlimits),criterion='center')
    
    temp=[]
    
    while sum(temp)<len(xy_known.T):
        hypercube = sampling(int(mul_fac*len(xy_known.T)))
        temp=[]
        hypercube_rounded=[]
        for item in hypercube:
            point=np.array([round_to_res(item[0],accuracy_e),round_to_res(item[1],accuracy_a)])
            hypercube_rounded.append(point)
            temp.append(np.any(np.all(xy.T==point,axis=1)))
        mul_fac+=0.1
    
    hypercube_temp=np.array([val for val, flag in zip(hypercube_rounded,temp) if flag])
    
    num=len(hypercube_temp)-len(xy_known.T)
    if num>0:
        random.shuffle(hypercube_temp)
        xy_rand=hypercube_temp[num:]
    else:
        xy_rand=np.array(hypercube_temp)

    return xy_rand.T

#####################################################################################################

def round_to_res(value,res):
    
    val=round(value/res)*res
    
    if res<1:
        val=np.around(val,str(res)[::-1].find('.'))
        
    return val

#####################################################################################################

def get_fit_points(xy_points,xy,z_fit,e_fit,z_func):
    
    z_fit_points=np.ones(len(xy_points.T))
    e_fit_points=np.ones(len(xy_points.T))
    z_func_points=np.ones(len(xy_points.T))
    
    for i in range(len(xy_points.T)):
        e_temp=(np.argwhere(xy[0]==xy_points.T[i][0]))
        a_temp=(np.argwhere(xy[1]==xy_points.T[i][1]))
        arg_temp=[item for item in e_temp if item in a_temp][0][0]
        z_fit_points[i]=z_fit[arg_temp]
        e_fit_points[i]=e_fit[arg_temp]
        z_func_points[i]=z_func[arg_temp]
    
    return z_fit_points,e_fit_points,z_func_points

###########################################################################################

def fit_gaussian1(x_data1,x_data2,range_data,bins):


    hist1, bin_edges1 = np.histogram(x_data1,bins=bins,range=range_data)
    hist1=hist1/sum(hist1)

    n1 = len(hist1)
    x_hist1=np.zeros((n1),dtype=float) 
    for i in range(n1):
        x_hist1[i]=(bin_edges1[i+1]+bin_edges1[i])/2

    y_hist1=hist1
    
    hist2, bin_edges2= np.histogram(x_data2,bins=bins,range=range_data)
    hist2=hist2/sum(hist2)

    n2 = len(hist2)
    x_hist2=np.zeros((n2),dtype=float) 
    for i in range(n2):
        x_hist2[i]=(bin_edges2[i+1]+bin_edges2[i])/2

    y_hist2=hist2

    #Calculating the Gaussian PDF values given Gaussian parameters and random variable X
    def gaus(X,C,X_mean,sigma):
        return C*np.exp(-(X-X_mean)**2/(2*sigma**2))

    mean1 = sum(x_hist1*y_hist1)/sum(y_hist1)                  
    sigma1 = sum(y_hist1*(x_hist1-mean1)**2)/sum(y_hist1) 

    #Gaussian least-square fitting process
    param_optimised1,param_covariance_matrix1 = curve_fit(gaus,x_hist1,y_hist1,\
                                                          p0=[max(y_hist1),mean1,sigma1],\
                                                          maxfev=500000)
    
    mean2 = sum(x_hist2*y_hist2)/sum(y_hist2)                  
    sigma2 = sum(y_hist2*(x_hist2-mean2)**2)/sum(y_hist2) 

    #Gaussian least-square fitting process
    param_optimised2,param_covariance_matrix2 = curve_fit(gaus,x_hist2,y_hist2,\
                                                          p0=[max(y_hist2),mean2,sigma2],\
                                                          maxfev=500000)
    
    
    return [param_optimised1[1],param_optimised1[2],param_optimised2[1],param_optimised2[2]]

#####################################################################################################################

def array_to_latex_table(input_array):
    print("\\begin{center}\n\\begin{tabularx}{1.0\\columnwidth}{|>{\\centering\\arraybackslash}X"+\
          "|>{\\raggedleft\\arraybackslash}X"*(len(input_array[0])-1)+\
      "|}\n\\hline&\multicolumn{2}{|c|}{Pseudodata} & \multicolumn{2}{|c|}{GP Datapoints}"+r"\\"+"\n\\hline")
    for i in range(len(input_array)):
        string=""
        j=0
        while j<len(input_array[i])-1:
            string+=str(input_array[i][j])+"&"
            j+=1
        string+=str(input_array[i][-1])+r"\\"
        print(string+"\n\\hline")
    print("\\end{tabularx}\n\\end{center}")
    
    return

####################################################################################################################

def coeff_pull_table():

    data_to_plot=pd.read_csv("pseudodata_results.csv")

    coeff_known_pull=data_to_plot["coeff_known_pull"].to_numpy()
    coeff_known_pull=np.array([np.fromstring(item.strip("[]"),sep=" ") for item in coeff_known_pull])
    coeff_GP_pull_rand=data_to_plot["coeff_GP_pull_rand"].to_numpy()
    coeff_GP_pull_rand=np.array([np.fromstring(item.strip("[]"),sep=" ") for item in coeff_GP_pull_rand])

    i=0
    coeff_pulls=[["Coefficient","Mean","Variance","Mean","Variance"]]

    array1=coeff_known_pull
    array2=coeff_GP_pull_rand

    while i<len(coeff_known_pull.T):
        l=legendre_orders[int(i/3)]
        
        a=fit_gaussian1(array1.T[i],array2.T[i],(-2,2),100)
        
        coeff_pulls.append(np.concatenate(([fr"$c_{l}$"], [f"{x:.2f}" for x in np.around(a, 2)])))
        
        
        b=fit_gaussian1(array1.T[i+1],array2.T[i+1],(-2,2),100)
        
        coeff_pulls.append(np.concatenate(([fr"$\mu_{l}$"], [f"{x:.2f}" for x in np.around(b, 2)])))
        
        c=fit_gaussian1(array1.T[i+2],array2.T[i+2],(-2,2),100)
        
        coeff_pulls.append(np.concatenate(([fr"$\sigma_"+str(l)+"^2$"], [f"{x:.2f}" for x in np.around(c, 2)])))

        i+=3

    coeff_pulls=np.array(coeff_pulls)

    array_to_latex_table(coeff_pulls)

    return


###################################################################################################################

def fit_pseudodata():

    xlimits=([coeff_limits]*len(legendre_orders))

    sampling = LHS(xlimits=np.array(xlimits),criterion='center')
    hypercube = sampling(hypercube_len)


    fraction=abs(coeff_limits[0]-coeff_limits[1])/(2*len(hypercube))
    decimal_places =str(fraction)[::-1].find('.')

    j=[]
    for i in range(len(hypercube)):
        j.append(i)

    data_dict=dict({"coefficients":j,"x_known":j,"y_known":j,"z_known":j,"e_known":j,"hyperparameter":j,\
                    "0.67std_percent_known":j,"1std_percent_known":j,"1.96std_percent_known":j,\
                    "0.67std_percent_fit_rand":j,"1std_percent_fit_rand":j,"1.96std_percent_fit_rand":j,\
                    "coeff_known_pull":j,"coeff_known":j,\
                    "coeff_GP_pull_rand":j,"coeff_GP_rand":j})

    data=pd.DataFrame(data_dict)

    for item in data_dict.keys():
        data[item]=data[item].astype("object")


    for k in range(len(hypercube)):
        print(k)
        coeff=[round(item,decimal_places) for item in hypercube[k]]
        
        coeff_all=[]
        
        for i in range(len(coeff)):
            gaus_mean=random.uniform(gaus_mean_limits[0],gaus_mean_limits[1])
            gaus_width=random.uniform(gaus_width_limits[0],gaus_width_limits[1])
            coeff_all.append(coeff[i])
            coeff_all.append(gaus_mean)
            coeff_all.append(gaus_width)  
            
        coeff_all=np.array(coeff_all)
        
        xy_known=assign_known_parameters(energy_limits,measured_angle_limits,\
                                        datapoints_energies_measured,datapoints_angles_measured)
            
        xy= create_convex_hull_boundary(xy_known,accuracy_e,accuracy_a)
        
        z_func,coeff_all=generate_pseudo_func(xy,coeff_all,legendre_orders)
        
        z_known,e_known,z_func_known=generate_pseudo_data(xy_known,xy,z_func,eff_count_limits)

        hyperpars=len_scale_opt(xy_known,z_known,e_known,sigma_check)
            
        z_fit,e_fit=GP(xy_known,z_known,e_known,xy,hyperpars)
        
        xy_rand=get_xy_rand(xy,xy_known,accuracy_a,accuracy_e)
        z_fit_rand,e_fit_rand,z_func_rand=get_fit_points(xy_rand,xy,z_fit,e_fit,z_func)
            
            
        coeff_known_pull,coeff_known=fit_data(xy_known,z_known,e_known,coeff_limits,\
                                                            gaus_mean_limits,gaus_width_limits,coeff_all)
        coeff_GP_pull_rand,coeff_GP_rand=fit_data(xy_rand,z_fit_rand,e_fit_rand,coeff_limits,\
                                                        gaus_mean_limits,gaus_width_limits,coeff_all)
            
        
        
        data.at[k,"coefficients"] = coeff_all 
        data.at[k,"x_known"] = xy_known[0]
        data.at[k,"y_known"] = xy_known[1]
        data.at[k,"z_known"] = z_known
        data.at[k,"e_known"] = e_known
        data.at[k,"hyperparameter"]=hyperpars

        data.at[k,"0.67std_percent_known"]=calculate_std_percent(z_known,z_func_known,e_known,0.67)
        data.at[k,"1std_percent_known"]=calculate_std_percent(z_known,z_func_known,e_known,1)
        data.at[k,"1.96std_percent_known"]=calculate_std_percent(z_known,z_func_known,e_known,1.96)
        
        data.at[k,"0.67std_percent_fit_rand"]=calculate_std_percent(z_fit_rand,z_func_rand,e_fit_rand,0.67)
        data.at[k,"1std_percent_fit_rand"]=calculate_std_percent(z_fit_rand,z_func_rand,e_fit_rand,1)
        data.at[k,"1.96std_percent_fit_rand"]=calculate_std_percent(z_fit_rand,z_func_rand,e_fit_rand,1.96) 
        
        data.at[k,"coeff_known_pull"]=coeff_known_pull
        data.at[k,"coeff_known"]=coeff_known
        
        data.at[k,"coeff_GP_pull_rand"]=coeff_GP_pull_rand
        data.at[k,"coeff_GP_rand"]=coeff_GP_rand

    data.to_csv('pseudodata_results.csv', index=False)

    return

################################################################################################################

def fit_to_func():
    def fix_array(array):
        return np.array([float(i) for i in array.strip("[]").split()])

    data=pd.read_csv("pseudodata_results.csv")

    r=random.randint(0,len(data))    
    data_example=data.loc[[r]]

    coeff_all=fix_array(data_example["coefficients"].to_numpy()[0])

    x_known=fix_array(data_example["x_known"].to_numpy()[0])
    y_known=fix_array(data_example["y_known"].to_numpy()[0])
    z_known=fix_array(data_example["z_known"].to_numpy()[0])
    e_known=fix_array(data_example["e_known"].to_numpy()[0])

    hyperpars=fix_array(data_example["hyperparameter"].to_numpy()[0])

    coeff_GP_rand=fix_array(data_example["coeff_GP_rand"].to_numpy()[0])
    coeff_known=fix_array(data_example["coeff_known"].to_numpy()[0])

    xy_known=np.column_stack((x_known,y_known)).T

    xy= create_convex_hull_boundary(xy_known,accuracy_e,accuracy_a)

    z_func,coeff=generate_pseudo_func(xy,coeff_all,legendre_orders)

    coeff_fit_GP_rand,coeff_GP1=generate_pseudo_func(xy,coeff_GP_rand,legendre_orders)
    coeff_fit_known,coeff_known1=generate_pseudo_func(xy,coeff_known,legendre_orders)

    z_fit,e_fit=GP(xy_known,z_known,e_known,xy,hyperpars)

    most_common=Counter(x_known).most_common(1)[0][0]

    y_known_temp=[]
    z_known_temp=[]
    e_known_temp=[]
    y_temp=[]
    y_gp=[]
    e_gp=[]
    y_fit=[]
    y_rand=[]
    y_func=[]

    for i in range(len(xy_known.T)):
        if xy_known.T[i][0]==most_common:
            y_known_temp.append(xy_known.T[i][1])
            z_known_temp.append(z_known[i])
            e_known_temp.append(e_known[i])

    for j in range(len(xy.T)):
        if xy.T[j][0]==most_common:
            y_temp.append(xy.T[j][1])
            y_func.append(z_func[j])
            y_gp.append(z_fit[j])
            e_gp.append(e_fit[j])
            y_rand.append(coeff_fit_GP_rand[j])
            y_fit.append(coeff_fit_known[j])


    plt.errorbar(y_known_temp,z_known_temp,yerr=e_known_temp,fmt="x",alpha=0.7,ecolor="black",color="black")
    plt.plot(y_temp,y_func,"y",label="Known Function",linestyle='-.')
    plt.plot(y_temp,y_fit,"r",label="Fit using known datapoints",linestyle='-.')
    plt.xlabel(r"$\cos\theta$")
    plt.grid()
    plt.legend()
    plt.savefig("known_fit.png")
    plt.close()

    plt.plot(y_temp,y_gp,label="GP")
    plt.plot(y_temp,y_func,"y",label="Known Function",linestyle='-.')
    plt.fill_between(y_temp,np.array(y_gp)-np.array(e_gp),np.array(y_gp)+np.array(e_gp),label="Standard Deviation",alpha=0.1)
    plt.errorbar(y_known_temp,z_known_temp,yerr=e_known_temp,fmt="x",alpha=0.7,ecolor="black",color="black")
    plt.xlabel(r"$\cos\theta$")
    plt.grid()
    plt.legend()
    plt.savefig("gp_fit.png")
    plt.close()

    plt.errorbar(y_known_temp,z_known_temp,yerr=e_known_temp,fmt="x",alpha=0.7,ecolor="black",color="black")
    plt.plot(y_temp,y_func,"y",label="Known Function",linestyle='-.')
    plt.plot(y_temp,y_fit,"r",label="Fit using known datapoints",linestyle='-.')
    plt.plot(y_temp,y_rand,label="Fit using GP datapoints",linestyle='-.')
    plt.xlabel(r"$\cos\theta$")
    plt.grid()
    plt.legend()
    plt.savefig("both_fit.png")
    plt.close()

    return

#################################################################################################################

energy_limits=[1.2,2]
angle_limits=[-1,1]
datapoints_energies_measured=4
datapoints_angles_measured=[4,10]
gaus_mean_limits=[1.4,1.8]       
gaus_width_limits=[0.25,0.75]
measured_angle_limits=[-0.85,0.85]   
coeff_limits=[-1,1]
eff_count_limits=[200,1000]
hypercube_len=100
legendre_orders=[0,1,2,3]
accuracy_a=0.01
accuracy_e=0.01
r_hat_tol=1.18
tau_tol=0.15

##########################################################################################################

#fit_pseudodata()

coeff_pull_table()

#fit_to_func()