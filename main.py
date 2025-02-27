import numpy as np
import pandas as pd
from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP,kernel_func

def create_GP():
    
    observable="Sigma"
    resolution=[0.01,0.01]
    plot=False

    xy_known,z_known,e_known=read_data(observable)

    len_scale=len_scale_opt(xy_known,z_known,e_known,plot)
    
    xy=fill_convex_hull(xy_known.T,resolution)
    
    y_fit,e_fit=GP(xy_known,z_known,e_known,xy.T,len_scale)

    print("Success")

    return

##############################################################################

def read_data(observable):

    df = pd.read_csv("g8K0SigResults.txt",sep=',', header=0)

    data=df.loc[df["obs"]==observable]

    xy_known=data[["Egamma","CosThetaK0"]].to_numpy().T
    z_known=data[["value"]].to_numpy().T[0]
    e_known=data[["std"]].to_numpy().T[0]
    
    return xy_known,z_known,e_known 

##############################################################################

create_GP()


