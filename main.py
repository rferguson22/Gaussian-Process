import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP,kernel_func

def create_GP():

    file_path,resolution,plot,out_file_name=read_yaml()
    
    xy_known,z_known,e_known=read_data(file_path)

    if len(resolution)!=len(xy_known):
        issue="Reading in the data at "+file_path+" showed a "+str(len(xy_known))+\
            "D problem but there are "+str(len(resolution))+\
            " resolution values listed. Please check how many dimensions your problem is."
        raise ValueError(issue)

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

def read_yaml():

    with open("options.yaml","r") as file:
        options=yaml.safe_load(file)


    file_path=options["file_name"]
    resolution=options["resolution"]
    plot=options["plot"]
    out_file_name=options["out_file_name"]


    file_path_check = Path(file_path)

    if not file_path_check.exists():
        raise ValueError("The input file does not exist. Please check the file path is correct in options.yaml")

    if not all(isinstance(item, (float,int)) for item in resolution):
        raise ValueError("The resolution list contains other types besides floats or integers."+\
                          " Please check that all items are floats or integers.")
    
    if plot is None:
        plot=False

    if out_file_name is None:
        original_file_path = Path(file_path)
        folder_path = original_file_path.parent
        out_file_name = folder_path / 'GP_results'

    return file_path,resolution,plot,out_file_name

##############################################################################

create_GP()


