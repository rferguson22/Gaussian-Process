import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
from pathlib import Path
from functools import reduce

from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP
from read_in import read_yaml

################################################################################

def process_experiment(x_known, y_known, e_known, resolution, dim_labels, filename, exp_idx, total_exps,\
                        MC_progress, MC_plotting, labels_out, out_file_name):
    
    """
    Perform Gaussian Process fitting for a single experiment and return the result as a DataFrame.
    """

    if x_known.shape[1] == 0:
        print(f"  Skipping experiment: no valid data points")
        return None

    len_scale = len_scale_opt(x_known, y_known, e_known, MC_progress, MC_plotting, labels_out, out_file_name)
    x_fit = fill_convex_hull(x_known.T, resolution)
    y_fit, e_fit = GP(x_known, y_known, e_known, x_fit.T, len_scale)

    df = pd.DataFrame(x_fit, columns=dim_labels)

    if total_exps == 1:
        quant_col = f"{filename}"
        err_col = f"{filename}_unc"
    else:
        quant_col = f"{filename}_exp{exp_idx}"
        err_col = f"{filename}_unc{exp_idx}"

    df[quant_col] = y_fit.flatten()
    df[err_col] = e_fit.flatten()

    return df

################################################################################

def write_individual_file(df, filename, out_path, total_exps, exp_idx):

    """
    Write a single experiment DataFrame to a csv file with the correct experiment index.
    """

    output_folder = out_path if out_path.is_dir() else out_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    if total_exps == 1:
        output_path = output_folder / f"{filename}_GP_results.txt"
    else:
        output_path = output_folder / f"{filename}_exp{exp_idx}_GP_results.txt"
    df.to_csv(output_path, index=False)
    print(f"Written individual output file: {output_path}")

    return

################################################################################

def write_grouped_file(file_dfs, filename, out_path, dim_labels):

    """
    Merge and write grouped experiment DataFrames from the same input file to a combined output file.
    """

    output_folder = out_path if out_path.is_dir() else out_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=dim_labels, how="outer"), file_dfs).fillna(float('inf'))
    output_path = output_folder / f"{filename}_GP_results.txt"
    merged_df.to_csv(output_path, index=False)

    print(f"Written grouped output file: {output_path}")

    return

################################################################################

def write_combined_file(experiment_dfs, out_path, dim_labels):

    """
    Merge all experiment DataFrames and write a single combined output csv file.
    """

    if str(out_path).endswith("/"):
        output_folder = out_path
        output_folder.mkdir(parents=True, exist_ok=True)
        combined_output_file = output_folder / "GP_results.txt"
    else:
        combined_output_file = out_path
        combined_output_file.parent.mkdir(parents=True, exist_ok=True)

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=dim_labels, how="outer"), experiment_dfs).fillna(float('inf'))
    merged_df.to_csv(combined_output_file, index=False)

    print(f"Combined results written to {combined_output_file}")

    return merged_df

################################################################################

def create_GP():

    """
    Main routine to perform GP fitting for all experiments and write output files based on configuration.
    """

    resolution, MC_progress, MC_plotting, out_file_name, labels, data_list, write_ind, group_exps = read_yaml()

    experiment_dfs = []
    num_dims = len(resolution)
    dim_labels = labels[:num_dims]
    out_path = Path(out_file_name) if out_file_name else Path("GP_results.txt")

    if write_ind:
        print("Writing individual output files")
        if group_exps:
            print("Writing experiments from the same file together")
        else:
            print("Writing experiments from the same file separately")
    else:
        print("Writing combined output file")

    for file_idx, (file_path, x_known_list, exp_pairs, labels_out) in enumerate(data_list, start=1):
        filename = Path(file_path).stem
        total_experiments = len(x_known_list)
        file_dfs = []

        for idx, (x_known, (y_known, e_known)) in enumerate(zip(x_known_list, exp_pairs), start=1):
            print(f"Processing experiment {idx}/{total_experiments} from file {file_idx}/{len(data_list)}: {filename}")

            df = process_experiment(x_known, y_known, e_known, resolution, dim_labels, filename, idx, total_experiments,\
                                     MC_progress, MC_plotting, labels_out, out_file_name)
            if df is None:
                continue

            experiment_dfs.append(df)
            file_dfs.append(df)

            if write_ind and not group_exps:
                write_individual_file(df, filename, out_path, total_experiments, idx)


        if write_ind and group_exps and file_dfs:
            write_grouped_file(file_dfs, filename, out_path, dim_labels)

    if not write_ind and experiment_dfs:
        return write_combined_file(experiment_dfs, out_path, dim_labels),num_dims

    if experiment_dfs:
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=dim_labels, how="outer"), experiment_dfs).fillna(float('inf'))
        return merged_df,num_dims
    else:
        print("No experiment data to return")
        return pd.DataFrame(),num_dims





