import os

cores_to_use = 2
os.environ["OMP_NUM_THREADS"] = str(cores_to_use)
os.environ["OPENBLAS_NUM_THREADS"] = str(cores_to_use)
os.environ["MKL_NUM_THREADS"] = str(cores_to_use)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores_to_use)
os.environ["NUMEXPR_NUM_THREADS"] = str(cores_to_use)
os.environ["MKL_DYNAMIC"] = "FALSE"


import math
import numpy as np
import json

from numpy import random
from numpy.polynomial.legendre import legval

from scipy.optimize import curve_fit
from scipy.special import legendre
from scipy.stats.distributions import norm
from scipy.stats import poisson
import scipy
import time

from iminuit import Minuit
from iminuit.cost import LeastSquares

import matplotlib.pyplot as plt

import math
import pandas as pd 
from smt.sampling_methods import LHS

from collections import Counter
import time  

from find_len_scales import len_scale_ks,len_scale_sigma,calculate_std_percent,calculate_pull,sigma_check,sigma_to_percent,\
    sigma_check_old,find_nll
from convex_hull import fill_convex_hull,round_to_res
from GP_func import GP

from threadpoolctl import threadpool_limits


#############################################################################################################

def create_convex_hull_boundary(points,accuracy_a):

    '''
    Creates the boundary assuming the binning edges need to be taken into account
    '''

    
    points = np.sort(points)
    decimal_places_a = str(accuracy_a)[::-1].find(".")

    a_min, a_max = points[0], points[-1]
    spacing = a_max - a_min

    left = round(a_min - 0.5 * spacing, decimal_places_a)
    right = round(a_max + 0.5 * spacing, decimal_places_a)

    left = max(left, -1)
    right = min(right, 1)
    
    x=np.arange(left,right+accuracy_a,accuracy_a)
    x=np.array([round_to_res(item,accuracy_a) for item in x])
    
    return x.T

##############################################################################################

def assign_known_parameters(measured_angle_limits,datapoints_angles_measured):
    

    '''
    Generates known kinematic points
    '''

    x_known=[]

    angles_range=abs(measured_angle_limits[0]-measured_angle_limits[1])/datapoints_angles_measured
    
    for i in range(datapoints_angles_measured):

        angle_temp=math.floor(random.uniform((measured_angle_limits[0]+(i*angles_range)),\
                                        (measured_angle_limits[0]+((i+1)*angles_range)))*100)/100
        
        x_known.append(angle_temp)

        
    return np.array(x_known)

##############################################################################################################

def generate_pseudo_func(x,coeff,legendre_orders):

    '''
    Generates pseduo function
    '''
    
    y_func = np.zeros_like(x, dtype=float)
    
    for c, order in zip(coeff, legendre_orders):
        poly_coeffs = [0] * order + [1]  # e.g., order=2 -> [0,0,1]
        y_func += c * legval(x, poly_coeffs)
    
    # scaling step (same as before)
    xs = np.linspace(0.6, 0.9, 1000)
    a = xs * (1 - xs)
    p = a / np.sum(a)
    scale = np.random.choice(xs, p=p) / max(abs(y_func))
    y_func *= scale
    coeff = np.array(coeff) * scale

    return y_func, coeff

#################################################################################################################

def generate_pseudo_data(x_known,x,y_func,eff_count_limits):

    '''
    Generates the pseduodata value
    '''

    y_known,e_known,y_func_known=[],[],[]   
    

    for i in range(len(x_known)):

        arg=np.argwhere(x_known[i]==x)
           
        a=y_func[arg]

        effective_count= np.random.uniform(low=eff_count_limits[0], high=eff_count_limits[1])

        n_plus=0.5*effective_count*(1+a)
        n_minus=0.5*effective_count*(1-a)

        N_plus=poisson.rvs(n_plus)
        N_minus=poisson.rvs(n_minus)

        y=(N_plus-N_minus)/(N_plus+N_minus)
        e=(2/((N_plus+N_minus)**2))*np.sqrt(N_plus*N_minus*(N_plus+N_minus))


        y_known.append(y)
        e_known.append(e)
        y_func_known.append(a)
            

    y_known=np.array(y_known)
    e_known=np.array(e_known) 
    y_func_known=np.array(y_func_known)
    
        
    return y_known,e_known,y_func_known

######################################################################################################

def fit_data(x_points,y_points,e_points,coeff_limits,coeff_all):

    '''
    Fits the functional form of the pseudodata surface
    '''

    initial_form=[np.mean(coeff_limits)]
    initial=np.array(initial_form*(len(coeff_all)))
    
    mi=Minuit(LeastSquares(x_points,y_points,e_points,model),initial)
    
    i=0
    
    mi.limits=coeff_limits
        
    mi.migrad()
    mi.hesse()
    
    coeff_guess=np.array(mi.values)
    coeff_guess_error=np.array(mi.errors)
    
    pull=calculate_pull(coeff_guess,coeff_all,coeff_guess_error)
        
    return pull,np.array(mi.values)

##############################################################################################################

def model(x_points,theta):

    '''
    Pseudodata function
    '''

    value = np.zeros_like(x_points, dtype=float)

    for coeff, order in zip(theta, legendre_orders):
        poly_coeffs = [0] * order + [1] 
        value += coeff * legval(x_points, poly_coeffs)

    return value

#############################################################################################################

def get_xy_rand(x,num_points,y_fit,e_fit,y_func):

    '''
    Samples kinematic points within convex hull
    '''

    bin_width=int(len(x)/num_points)

    index=[]

    for i in range(num_points):
        index.append(np.random.randint(i*bin_width,np.minimum((i+1)*bin_width,len(x)-1)))


    return x[index],y_fit[index],e_fit[index],y_func[index]

#####################################################################################################

def get_fit_points(xy_points,xy,z_fit,e_fit,z_func):

    '''
    Gets GP values from sampled points
    '''
    
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

def fit_gaussian(x_data1,x_data2,range_data,bins):

    '''
    Fits Gaussian
    '''

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

def coeff_pull_table():

    '''
    Gets the pull of the fitted coefficients
    '''

    data_to_plot=pd.read_csv("pseudodata_results.csv")

    coeff_known_pull=data_to_plot["coeff_known_pull"].to_numpy()
    coeff_known_pull=np.array([np.fromstring(item.strip("[]"),sep=" ") for item in coeff_known_pull])
    coeff_GP_pull_rand=data_to_plot["coeff_GP_pull_rand"].to_numpy()
    coeff_GP_pull_rand=np.array([np.fromstring(item.strip("[]"),sep=" ") for item in coeff_GP_pull_rand])

    i=0
    coeff_pulls=[]

    array1=coeff_known_pull
    array2=coeff_GP_pull_rand

    while i<len(coeff_known_pull.T):
        l=legendre_orders[int(i/3)]
        
        a=fit_gaussian(array1.T[i],array2.T[i],(-2,2),100)
        
        coeff_pulls.append(a)
        
        
        b=fit_gaussian(array1.T[i+1],array2.T[i+1],(-2,2),100)
        
        coeff_pulls.append(b)
        
        c=fit_gaussian(array1.T[i+2],array2.T[i+2],(-2,2),100)
        
        coeff_pulls.append(c)

        i+=3

    coeff_pulls=np.array(coeff_pulls)

    print(coeff_pulls)

    return coeff_pulls

###################################################################################################################

def load_pseudodata(filename,k):
    data = pd.read_csv(filename)

    # JSON decode every column
    for col in data.columns:
        data[col] = data[col].apply(json.loads)

    # Convert numeric arrays into numpy arrays
    numeric_cols = ["coefficients", "x_known", "y_known", "e_known", "y_func_known"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].apply(np.array)

    data_example = data.loc[[k]]

    coeff_all    = data_example["coefficients"].to_numpy()[0]
    x_known      = data_example["x_known"].to_numpy()[0]
    y_known      = data_example["y_known"].to_numpy()[0]
    e_known      = data_example["e_known"].to_numpy()[0]
    y_func_known = data_example["y_func_known"].to_numpy()[0]

    
    return coeff_all,x_known,y_known,e_known,y_func_known

########################################################################################################

def fit_pseudodata():
    '''
    Fits pseudodata
    '''
    xlimits = ([coeff_limits] * len(legendre_orders))
    sampling = LHS(xlimits=np.array(xlimits), criterion='center')
    hypercube = sampling(hypercube_len)

    fraction = abs(coeff_limits[0] - coeff_limits[1]) / (2 * len(hypercube))
    decimal_places = str(fraction)[::-1].find('.')

    j = list(range(len(hypercube)))

    data_dict = dict({
        "hyperparameter": j, "loss_score":j,"0.67std_percent_known": j, "1std_percent_known": j,
        "1.96std_percent_known": j, "0.67std_percent_fit_rand": j, "1std_percent_fit_rand": j,
        "1.96std_percent_fit_rand": j, "coeff_known_pull": j, "coeff_known": j,
        "coeff_GP_pull_rand": j, "coeff_GP_rand": j,
        "runtime_seconds": j  
    })

    data = pd.DataFrame(data_dict)

    for item in data_dict.keys():
        data[item] = data[item].astype("object")


    for k in range(hypercube_len):
        print(k)

        coeff_all,x_known,y_known,e_known,y_func_known=load_pseudodata("pseudodata_inputs50.csv",k)

        x = create_convex_hull_boundary(x_known,accuracy_a)
        y_func, coeff_all = generate_pseudo_func(x, coeff_all, legendre_orders)

        '''
        lengths=np.array([0.1,0.1])
        loss_old=sigma_check_old(lengths,xy_known,y_known,e_known)

        sigma_vals = np.linspace(0.001, 3, 1000)
        expected_percents = sigma_to_percent(sigma_vals)
        loss_new=sigma_check(lengths, xy_known, y_known, e_known,sigma_vals,expected_percents)

        print("old:",loss_old)
        print("new:",loss_new)

        return

        '''

        x_known=x_known.reshape((1,len(x_known)))

        start_time = time.process_time()
        #hyperpars, score = len_scale_ks(xy_known, z_known, e_known, True, False, None, "")
        #output_file="pseudodata_ks.csv"
        hyperpars, score = len_scale_sigma(x_known, y_known, e_known, True, False, None, "")
        output_file = "pseudodata_sigma_quad.csv"
        #hyperpars,score=find_nll(x_known.T,y_known,e_known)
        #output_file = "pseudodata_nll10.csv"
        end_time = time.process_time()

        y_fit, e_fit = GP(x_known, y_known, e_known, x, hyperpars)

        x_rand,y_fit_rand,e_fit_rand,y_func_rand = get_xy_rand(x, len(x_known.T),y_fit,e_fit,y_func)

        coeff_known_pull, coeff_known = fit_data(x_known.flatten(), y_known, e_known, coeff_limits,coeff_all)

        coeff_GP_pull_rand, coeff_GP_rand = fit_data(x_rand,y_fit_rand, e_fit_rand, coeff_limits,coeff_all)

        percent95=calculate_std_percent(y_fit_rand, y_func_rand, e_fit_rand, 1.96)

        data.at[k, "hyperparameter"] = hyperpars
        data.at[k, "loss_score"]=score

        y_func_known = np.squeeze(y_func_known)

        data.at[k, "0.67std_percent_known"] = calculate_std_percent(y_known, y_func_known, e_known, 0.67)
        data.at[k, "1std_percent_known"] = calculate_std_percent(y_known, y_func_known, e_known, 1)
        data.at[k, "1.96std_percent_known"] = calculate_std_percent(y_known, y_func_known, e_known, 1.96)

        data.at[k, "0.67std_percent_fit_rand"] = calculate_std_percent(y_fit_rand, y_func_rand, e_fit_rand, 0.67)
        data.at[k, "1std_percent_fit_rand"] = calculate_std_percent(y_fit_rand, y_func_rand, e_fit_rand, 1)
        data.at[k, "1.96std_percent_fit_rand"] = calculate_std_percent(y_fit_rand, y_func_rand, e_fit_rand, 1.96)

        print("95%:",percent95)

        data.at[k, "coeff_known_pull"] = coeff_known_pull
        data.at[k, "coeff_known"] = coeff_known

        data.at[k, "coeff_GP_pull_rand"] = coeff_GP_pull_rand
        data.at[k, "coeff_GP_rand"] = coeff_GP_rand

        data.at[k, "runtime_seconds"] = end_time - start_time  

        plotting_check(x_known.flatten(),y_known,e_known,x,y_func,y_fit,e_fit,x_rand,y_fit_rand,k,percent95)

    data.to_csv(output_file, index=False)

    return

################################################################################################################

def fix_array(array):
        return np.array([float(i) for i in array.strip("[]").split()])

############################################################################################################

def fit_to_func():

    '''
    Generates pseudodata testing graphs
    '''
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
    plt.savefig("./pseudodata_graphs/known_fit.png")
    plt.close()

    plt.plot(y_temp,y_gp,label="GP")
    plt.plot(y_temp,y_func,"y",label="Known Function",linestyle='-.')
    plt.fill_between(y_temp,np.array(y_gp)-np.array(e_gp),np.array(y_gp)+np.array(e_gp),label="Standard Deviation",alpha=0.1)
    plt.errorbar(y_known_temp,z_known_temp,yerr=e_known_temp,fmt="x",alpha=0.7,ecolor="black",color="black")
    plt.xlabel(r"$\cos\theta$")
    plt.grid()
    plt.legend()
    plt.savefig("./pseudodata_graphs/gp_fit.png")
    plt.close()

    plt.errorbar(y_known_temp,z_known_temp,yerr=e_known_temp,fmt="x",alpha=0.7,ecolor="black",color="black")
    plt.plot(y_temp,y_func,"y",label="Known Function",linestyle='-.')
    plt.plot(y_temp,y_fit,"r",label="Fit using known datapoints",linestyle='-.')
    plt.plot(y_temp,y_rand,label="Fit using GP datapoints",linestyle='-.')
    plt.xlabel(r"$\cos\theta$")
    plt.grid()
    plt.legend()
    plt.savefig("./pseudodata_graphs/both_fit.png")
    plt.close()

    return

######################################################################################

def print_std():

    data=pd.read_csv("pseudodata_sigma_quad.csv")

    print(f"0.67 known: \t{np.mean(data['0.67std_percent_known'].to_numpy())}")
    print(f"0.67 GP: \t{np.mean(data['0.67std_percent_fit_rand'].to_numpy())}\n")

    print(f"1 known: \t{np.mean(data['1std_percent_known'].to_numpy())}")
    print(f"1 GP: \t\t{np.mean(data['1std_percent_fit_rand'].to_numpy())}\n")

    print(f"1.96 known: \t{np.mean(data['1.96std_percent_known'].to_numpy())}")
    print(f"1.96 GP: \t{np.mean(data['1.96std_percent_fit_rand'].to_numpy())}\n")

    return
        
#################################################################################################################

def gen_pseudo_data():

    print("Generating pseudodata")


    # Create LHS sampling
    xlimits = ([coeff_limits] * len(legendre_orders))
    sampling = LHS(xlimits=np.array(xlimits), criterion='center')
    hypercube = sampling(hypercube_len)

    fraction = abs(coeff_limits[0] - coeff_limits[1]) / (2 * len(hypercube))
    decimal_places = str(fraction)[::-1].find('.')

    j = list(range(len(hypercube)))

    data_dict = {
        "coefficients": j,
        "x_known": j,
        "y_known": j,
        "e_known": j,
        "y_func_known": j,
    }

    data = pd.DataFrame(data_dict)

    # make sure all columns are object dtype
    for item in data_dict.keys():
        data[item] = data[item].astype("object")

    for k in range(len(hypercube)):
        print(k)

        coeff = np.array([round(item, decimal_places) for item in hypercube[k]])

        x_known = assign_known_parameters(measured_angle_limits, datapoints_angles_measured)

        x = create_convex_hull_boundary(x_known,accuracy_a)
        y_func, coeff = generate_pseudo_func(x, coeff, legendre_orders)
        y_known, e_known, y_func_known = generate_pseudo_data(x_known,x, y_func, eff_count_limits)

        data.at[k, "coefficients"] = coeff.tolist()
        data.at[k, "x_known"] = x_known.tolist()
        data.at[k, "y_known"] = y_known.tolist()
        data.at[k, "e_known"] = e_known.tolist()
        data.at[k, "y_func_known"] = y_func_known.tolist()
       
    for item in data.columns:
        data[item] = data[item].apply(lambda x: json.dumps(x))

    data.to_csv("pseudodata_inputs.csv", index=False)

    return

################################################################################################################

def plot_comparison():

    file1 = "pseudodata_ks.csv"
    file2 = "pseudodata_sigma.csv"  
    directory="./Graphs/"

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    
    def extract_value(s,i):
        values = s.strip('[]').split()
        return float(values[i])

    df1['first_len_scale']  = df1['hyperparameter'].apply(lambda x: extract_value(x, 0))
    df2['first_len_scale']  = df2['hyperparameter'].apply(lambda x: extract_value(x, 0))

    df1['second_len_scale'] = df1['hyperparameter'].apply(lambda x: extract_value(x, 1))
    df2['second_len_scale'] = df2['hyperparameter'].apply(lambda x: extract_value(x, 1))

    first_diff = df1["first_len_scale"] - df2["first_len_scale"]
    second_diff = df1["second_len_scale"] - df2["second_len_scale"]
    runtime_diff = df1["runtime_seconds"] - df2["runtime_seconds"]

    plt.hist(first_diff, bins=7)
    plt.title("Difference in First Length Scale (KS - Sigma)")
    plt.savefig(directory+"first_diff.png")
    plt.close()

    plt.hist(second_diff, bins=7)
    plt.title("Difference in Second Length Scale (KS - Sigma)")
    plt.savefig(directory+"second_diff.png")
    plt.close()

    plt.hist(runtime_diff, bins=7)
    plt.title("Difference in Runtime (KS - Sigma)")
    plt.savefig(directory+"runtime_diff.png")

    return

#####################################################################################################################

def plotting_check(x_known,y_known,e_known,x,y_func,y_fit,e_fit,x_rand,y_rand,graph_num,percent95):

    plt.plot(x,y_fit,label="GP")
    plt.plot(x,y_func,"y",label="Known Function",linestyle='-.')
    plt.fill_between(x,y_fit-e_fit,y_fit+e_fit,label="Standard Deviation",alpha=0.1)
    plt.errorbar(x_known,y_known,yerr=e_known,fmt="x",alpha=0.7,ecolor="black",color="black",label="Known Points")
    plt.scatter(x_rand,y_rand,label="Random Points")
    plt.title("2sigma of random points="+str(percent95))
    plt.savefig("pseudodata_graphs/Graph"+str(graph_num)+".png")
    plt.close()

    return

##################################################################################################################

angle_limits=[-1,1]
datapoints_angles_measured=50
measured_angle_limits=[-0.85,0.85]   
coeff_limits=[-1,1]
eff_count_limits=[200,1000]
hypercube_len=100
legendre_orders=[0,1,2,3]
accuracy_a=0.01
r_hat_tol=1.18
tau_tol=0.15

##########################################################################################################

#gen_pseudo_data()

#fit_pseudodata()

#plot_comparison()

#coeff_pull_table()

#fit_to_func()

print_std()
