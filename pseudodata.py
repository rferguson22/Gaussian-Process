import numpy as np
from numpy import random

from scipy.optimize import curve_fit
from scipy.special import legendre
from scipy.stats.distributions import norm
from scipy.stats import poisson
import scipy

from iminuit import Minuit
from iminuit.cost import LeastSquares

import matplotlib.pyplot as plt

import math
import pandas as pd 
from smt.sampling_methods import LHS

from collections import Counter

from find_len_scales import len_scale_opt,calculate_std_percent,calculate_pull
from convex_hull import fill_convex_hull,round_to_res
from GP_func import GP

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

    return


###################################################################################################################

def fit_pseudodata():

    '''
    Fits pseudodata
    '''

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

        hyperpars=len_scale_opt(xy_known,z_known,e_known,False,False,"","")
            
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

    '''
    Generates pseudodata testing graphs
    '''

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

#coeff_pull_table()

fit_to_func()