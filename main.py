import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP
import time

def create_GP():

    timings = {}
    overall_start = time.time()

    start = time.time()
    file_path, resolution, MC_progress, MC_plotting, out_file_name, labels = read_yaml()
    timings['read_yaml'] = time.time() - start

    start = time.time()
    x_known, y_known, e_known, labels = read_data(file_path, labels)
    timings['read_data'] = time.time() - start

    if len(resolution) != len(x_known):
        issue = (
            "Reading in the data at " + file_path +
            " showed a " + str(len(x_known)) +
            "D problem but there are " + str(len(resolution)) +
            " resolution values listed. Please check how many dimensions your problem is."
        )
        raise ValueError(issue)

    start = time.time()
    len_scale = len_scale_opt(x_known, y_known, e_known, MC_progress, MC_plotting, labels, out_file_name)
    timings['len_scale_opt'] = time.time() - start

    start = time.time()
    x_fit = fill_convex_hull(x_known.T, resolution)
    timings['fill_convex_hull'] = time.time() - start

    start = time.time()
    y_fit, e_fit = GP(x_known, y_known, e_known, x_fit.T, len_scale)
    timings['GP'] = time.time() - start

    start = time.time()
    output_GP(x_fit, y_fit, e_fit, out_file_name, labels)
    timings['output_GP'] = time.time() - start

    timings['total_time'] = time.time() - overall_start

    return timings

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

    x_known=data.iloc[:,:-2].to_numpy().T
    y_known=data.iloc[:,-2].to_numpy().T
    e_known=data.iloc[:,-1].to_numpy().T
    
    return x_known,y_known,e_known,labels

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

def output_GP(x_fit,y_fit,e_fit,out_file_name,labels):

    '''
    Outputs the GP fits
    '''

    data=np.vstack([x_fit.T,y_fit.T,e_fit.T])

    df=pd.DataFrame(data.T,columns=labels)

    df.to_csv(out_file_name,index=False)

    return

################################################################################

df = pd.DataFrame()

for i in range(100):
    print(f"Run {i + 1}/100")
    try:
        result = create_GP()
        result['run'] = i + 1
    except Exception as e:
        result = {'run': i + 1, 'error': str(e)}
        print(f"Error during run {i + 1}: {e}")
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

df.to_csv("timing_log.csv", index=False)
print("Timing log saved to timing_log.csv")


