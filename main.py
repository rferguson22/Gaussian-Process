import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
from pathlib import Path
import yaml
from functools import reduce

from GP_fit import create_GP
from calc_prob_surf import generate_prob_surf
from read_in import expand_file_paths,check_data

###########################################################################################

def load_and_merge_gp_results(file_entries, resolution, labels):

    '''
    Load GP result files, process their data, and merge all into a single DataFrame based on resolution label
    '''
    
    file_paths = expand_file_paths(file_entries)
    data_list = check_data(file_paths, resolution, labels)

    if not data_list:
        print("No files loaded.")
        return pd.DataFrame(), len(resolution)

    dim_labels = None
    all_dfs = []

    for file_path, x_known_list, exp_pairs, labels_out in data_list:
        if dim_labels is None:
            dim_labels = labels_out[:len(resolution)]

        for exp_idx, (x_known, (y_known, e_known)) in enumerate(zip(x_known_list, exp_pairs), start=1):
            df = pd.DataFrame(x_known.T, columns=dim_labels)
            quant_col = f"{Path(file_path).stem}_exp{exp_idx}"
            err_col = f"{quant_col}_unc"
            df[quant_col] = y_known
            df[err_col] = e_known
            all_dfs.append(df)

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=dim_labels, how="outer"), all_dfs).fillna(float('inf'))

    print("All GP results loaded and merged.")

    return merged_df, len(resolution)


#############################################################################################

def main_gp_flow():

    '''
    Control flow for GP fitting or loading and merging GP results based on options.yaml configuration
    '''

    with open("options.yaml", "r") as f:
        options = yaml.safe_load(f)

    gp_fit = options.get("gp_fit", True)
    run_prob_surf = options.get("run_prob_surf", True)
    resolution = options["resolution"]
    labels = options.get("labels", None)
    file_entries = options["file_name"]

    if gp_fit:
        df, ndims = create_GP()
    else:
        df, ndims = load_and_merge_gp_results(file_entries, resolution, labels)

    return df, ndims, run_prob_surf

#####################################################################################################################

def main():
    '''
    Main entry point that runs the GP workflow and optionally generates a probability surface based on options.yaml
    '''


    df, ndims, run_prob_surf = main_gp_flow()
    if run_prob_surf:
        print("Generating probability surface...")
        generate_prob_surf(df, ndims)
    else:
        print("Skipping probability surface generation as per options.yaml")

        return


########################################################################################################################

if __name__ == "__main__":
    main()

