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

from scipy.optimize import curve_fit
from scipy.special import legendre
from scipy.stats.distributions import norm
from scipy.stats import poisson
import scipy
import time

from iminuit import Minuit
from iminuit.cost import LeastSquares

import matplotlib.pyplot as plt
import seaborn as sns

import math
import pandas as pd 
from smt.sampling_methods import LHS

from collections import Counter
import time  

from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull,round_to_res
from GP_func import GP

from threadpoolctl import threadpool_limits


#############################################################################################################

def create_convex_hull_boundary(xy_known,accuracy_e,accuracy_a):

    '''
    Creates the boundary assuming the binning edges need to be taken into account
    '''

    
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
    
    xy=fill_convex_hull(points,[accuracy_e,accuracy_a])
    
    return xy.T

##############################################################################################

def assign_known_parameters(energy_limits,measured_angle_limits,\
                            datapoints_energies_measured,datapoints_angles_measured):
    

    '''
    Generates known kinematic points
    '''

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

    '''
    Generates pseduo function
    '''
    
    z_func=np.zeros(len(xy.T))

    for i in range(len(z_func)):
        j=0
        while j<len(coeff_all):
            order=legendre_orders[int(j/3)]
            b=(norm.pdf(xy[0][i],coeff_all[j+1],coeff_all[j+2])\
                        *coeff_all[j]*legendre(order)(xy[1][i]))
            z_func[i]+=b
            j+=3

    if max(abs(z_func))>1:
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

    '''
    Generates the pseduodata value
    '''

    z_known,e_known,z_func_known=[],[],[]   
    

    for i in range(len(xy_known.T)):
        
        e_temp=np.argwhere(xy_known.T[i][0]==xy[0])
        a_temp=np.argwhere(xy_known.T[i][1]==xy[1])
        coeff_temp=[item for item in a_temp if item in e_temp][0][0]
        
        a=z_func[coeff_temp]

        effective_count = np.random.uniform(low=eff_count_limits[0], high=eff_count_limits[1])
        N_total = poisson.rvs(effective_count)

        N_plus = np.random.binomial(N_total, (1 + a)/2)
        N_minus = N_total - N_plus

        z = (N_plus - N_minus) / N_total
        e = np.sqrt((1 - z**2) / N_total)

        z_known.append(z)
        e_known.append(e)
        z_func_known.append(a)
            

    z_known=np.array(z_known)
    e_known=np.array(e_known) 
    z_func_known=np.array(z_func_known)
    
        
    return z_known,e_known,z_func_known

######################################################################################################

def fit_data(xy_points,z_points,e_points,coeff_limits,gaus_mean_limits,gaus_width_limits,coeff_all):

    '''
    Fits the functional form of the pseudodata surface
    '''

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

    '''
    Pseudodata function
    '''

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

    '''
    Samples kinematic points within convex hull
    '''

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

#########################################################################################################

def load_pseudodata(filename,k):
    data = pd.read_csv(filename)

    # JSON decode every column
    for col in data.columns:
        data[col] = data[col].apply(json.loads)

    # Convert numeric arrays into numpy arrays
    numeric_cols = ["coefficients", "x_known", "y_known", "z_known", "e_known", "z_func_known"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].apply(np.array)

    data_example = data.loc[[k]]

    coeff_all    = data_example["coefficients"].to_numpy()[0]
    x_known      = data_example["x_known"].to_numpy()[0]
    y_known      = data_example["y_known"].to_numpy()[0]
    z_known      = data_example["z_known"].to_numpy()[0]
    e_known      = data_example["e_known"].to_numpy()[0]
    z_func_known = data_example["z_func_known"].to_numpy()[0]

    
    return coeff_all,x_known,y_known,z_known,e_known,z_func_known

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
        "hyperparameter": j, "loss_score":j,"upper_bounds":j,"lower_bounds":j,\
        "0.67std_percent_known": j, "1std_percent_known": j,"1.96std_percent_known": j,\
        "0.67std_percent_fit_rand": j, "1std_percent_fit_rand": j,"1.96std_percent_fit_rand":j,\
        "coeff_known_pull": j, "coeff_known": j,"coeff_GP_pull_rand": j, "coeff_GP_rand": j,\
        "cpu_runtime": j,"wall_runtime":j  
    })

    data = pd.DataFrame(data_dict)

    for item in data_dict.keys():
        data[item] = data[item].astype("object")


    for k in range(hypercube_len):
        print(k)

        coeff_all,x_known,y_known,z_known,e_known,z_func_known=load_pseudodata("pseudodata_inputs.csv",k)


        xy_known=np.stack((x_known,y_known))

        xy = create_convex_hull_boundary(xy_known, accuracy_e, accuracy_a)
        z_func, coeff_all = generate_pseudo_func(xy, coeff_all, legendre_orders)

        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        hyperpars,score = len_scale_opt(xy_known, z_known, e_known,True)
        end_cpu = time.process_time()
        end_wall = time.perf_counter()

        print("CPU time used: ", end_cpu - start_cpu)
        print("Wall-clock time: ", end_wall - start_wall)

        z_fit, e_fit = GP(xy_known, z_known, e_known, xy, hyperpars)

        xy_rand = get_xy_rand(xy, xy_known, accuracy_a, accuracy_e)
        z_fit_rand, e_fit_rand, z_func_rand = get_fit_points(xy_rand, xy, z_fit, e_fit, z_func)

        coeff_known_pull, coeff_known = fit_data(
            xy_known, z_known, e_known, coeff_limits,
            gaus_mean_limits, gaus_width_limits, coeff_all
        )

        coeff_GP_pull_rand, coeff_GP_rand = fit_data(
            xy_rand, z_fit_rand, e_fit_rand, coeff_limits,
            gaus_mean_limits, gaus_width_limits, coeff_all
        )

        data.at[k, "hyperparameter"] = hyperpars
        data.at[k, "loss_score"]=score

        data.at[k, "0.67std_percent_known"] = calculate_std_percent(z_known, z_func_known, e_known, 0.67)
        data.at[k, "1std_percent_known"] = calculate_std_percent(z_known, z_func_known, e_known, 1)
        data.at[k, "1.96std_percent_known"] = calculate_std_percent(z_known, z_func_known, e_known, 1.96)

        data.at[k, "0.67std_percent_fit_rand"] = calculate_std_percent(z_fit_rand, z_func_rand, e_fit_rand, 0.67)
        data.at[k, "1std_percent_fit_rand"] = calculate_std_percent(z_fit_rand, z_func_rand, e_fit_rand, 1)
        data.at[k, "1.96std_percent_fit_rand"] = calculate_std_percent(z_fit_rand, z_func_rand, e_fit_rand, 1.96)

        data.at[k, "coeff_known_pull"] = coeff_known_pull
        data.at[k, "coeff_known"] = coeff_known

        data.at[k, "coeff_GP_pull_rand"] = coeff_GP_pull_rand
        data.at[k, "coeff_GP_rand"] = coeff_GP_rand

        data.at[k, "cpu_runtime"] = end_cpu - start_cpu  
        data.at[k, "wall_runtime"] = end_wall - start_wall  

    data.to_csv("pseudodata_output.csv", index=False)

    return

################################################################################################################

def fix_array(array):
    cleaned = array.strip("[]").replace(",", " ")
    return np.array([float(i) for i in cleaned.split()])

####################################################################################################################

def gen_pseudo_data():

    print("Generating pseudodata")

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
        "z_known": j,
        "e_known": j,
        "z_func_known": j,
    }

    data = pd.DataFrame(data_dict)

    for item in data_dict.keys():
        data[item] = data[item].astype("object")

    for k in range(len(hypercube)):
        print(k)

        coeff = [round(item, decimal_places) for item in hypercube[k]]
        coeff_all = []

        for i in range(len(coeff)):
            gaus_mean = random.uniform(gaus_mean_limits[0], gaus_mean_limits[1])
            gaus_width = random.uniform(gaus_width_limits[0], gaus_width_limits[1])
            coeff_all.append(coeff[i])
            coeff_all.append(gaus_mean)
            coeff_all.append(gaus_width)  

        coeff_all = np.array(coeff_all)

        xy_known = assign_known_parameters(
            energy_limits, measured_angle_limits,
            datapoints_energies_measured, datapoints_angles_measured
        )

        xy = create_convex_hull_boundary(xy_known, accuracy_e, accuracy_a)
        z_func, coeff_all = generate_pseudo_func(xy, coeff_all, legendre_orders)
        z_known, e_known, z_func_known = generate_pseudo_data(
            xy_known, xy, z_func, eff_count_limits
        )

        data.at[k, "coefficients"] = coeff_all.tolist()
        data.at[k, "x_known"] = xy_known[0].tolist()
        data.at[k, "y_known"] = xy_known[1].tolist()
        data.at[k, "z_known"] = z_known.tolist()
        data.at[k, "e_known"] = e_known.tolist()
        data.at[k, "z_func_known"] = z_func_known.tolist()
       
    for item in data.columns:
        data[item] = data[item].apply(lambda x: json.dumps(x))

    data.to_csv("pseudodata_inputs.csv", index=False)

    return

################################################################################################################

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

##########################################################################################################

#gen_pseudo_data()

fit_pseudodata()

