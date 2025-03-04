import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP,kernel_func

def create_GP():

    file_path,resolution,MC_progress,MC_plotting,out_file_name,labels=read_yaml()
    
    xy_known,z_known,e_known,labels=read_data(file_path,labels)

    if len(resolution)!=len(xy_known):
        issue="Reading in the data at "+file_path+" showed a "+str(len(xy_known))+\
            "D problem but there are "+str(len(resolution))+\
            " resolution values listed. Please check how many dimensions your problem is."
        raise ValueError(issue)

    len_scale=len_scale_opt(xy_known,z_known,e_known,MC_progress,MC_plotting,labels,out_file_name)
    
    xy=fill_convex_hull(xy_known.T,resolution)
    
    y_fit,e_fit=GP(xy_known,z_known,e_known,xy.T,len_scale)

    print("Success")

    return

##############################################################################

def generate_labels(num_columns):

    '''
    Generates default labels for the GP_output file. 
    '''

    labels= [f'dim{i+1}' for i in range(num_columns-2)]

    labels.extend(['quantity', 'error'])

    return labels

##############################################################################

def read_csv(file_path):

    '''
    Reads in csv file, checking if there a header.
    '''
    
    df_no_header = pd.read_csv(file_path, header=None)
    
    first_row = df_no_header.iloc[0]
    
    if all(isinstance(val, str) for val in first_row):  
        df = pd.read_csv(file_path) 
    else:
        labels=generate_labels(len(df_no_header.columns))
        df = pd.read_csv(file_path, names=labels)
    
    return df

##############################################################################

def read_data(file_path,labels):

    '''
    Reads in known datapoints.
    '''

    data = read_csv(file_path)

    if labels is None:
        labels=data.columns

    if len(labels)!=len(data.columns):
        issue="Expected "+str(len(data.columns))+ " column names in labels but received "+str(len(labels))
        raise ValueError(issue)

    xy_known=data.iloc[:,:-2].to_numpy().T
    z_known=data.iloc[:,-2].to_numpy().T
    e_known=data.iloc[:,-1].to_numpy().T
    
    return xy_known,z_known,e_known,labels

##############################################################################

def read_yaml():

    with open("options.yaml","r") as file:
        options=yaml.safe_load(file)


    file_path=options["file_name"]
    resolution=options["resolution"]
    MC_progress=options["MC_progress"]
    MC_plotting=options["MC_plotting"]
    out_file_name=options["out_file_name"]
    labels=options["labels"]


    file_path_check = Path(file_path)

    if not file_path_check.exists():
        raise ValueError("The input file does not exist. Please check the file path is correct in options.yaml")

    if not all(isinstance(item, (float,int)) for item in resolution):
        raise ValueError("The resolution list contains other types besides floats or integers."+\
                          " Please check that all items are floats or integers.")
    
    if MC_progress is None:
        MC_progress=False

    if MC_plotting is None:
        MC_plotting=False

    if out_file_name is None:
        original_file_path = Path(file_path)
        folder_path = original_file_path.parent
        out_file_name = folder_path / 'GP_results.txt'

    return file_path,resolution,MC_progress,MC_plotting,out_file_name,labels

##############################################################################

create_GP()


