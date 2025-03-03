import numpy as np
import pandas as pd
import yaml
from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP,kernel_func

def create_GP():
    
    with open("options.yaml","r") as file:
        data=yaml.safe_load(file)


    file_path=data["file_name"]
    resolution=data["resolution"]
    plot=False

    xy_known,z_known,e_known=read_data(file_path)

    len_scale=len_scale_opt(xy_known,z_known,e_known,plot)
    
    xy=fill_convex_hull(xy_known.T,resolution)
    
    y_fit,e_fit=GP(xy_known,z_known,e_known,xy.T,len_scale)

    print("Success")

    return

##############################################################################

def read_data(file_path):

    data = pd.read_csv(file_path,sep=',', header=0)

    xy_known=data.iloc[:,:-2].to_numpy().T
    z_known=data.iloc[:,-2].to_numpy().T
    e_known=data.iloc[:,-1].to_numpy().T
    
    return xy_known,z_known,e_known 

##############################################################################

create_GP()


