import os
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from collections import defaultdict

from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP
from read_in import read_yaml

################################################################################

def write_combined_output(experiment_dfs, out_file_name, dim_labels):
    
    """
    Merge multiple experiment DataFrames and write the combined result to a single output csv file.
    """
    if not experiment_dfs:
        print("No data to write in combined output.")
        return

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=dim_labels, how="outer"), experiment_dfs)
    merged_df = merged_df.fillna(float('inf'))
    merged_df.to_csv(out_file_name, index=False)

    print(f"Combined results written to {out_file_name}")

    return

################################################################################

def write_individual_outputs(experiment_dfs, out_folder, dim_labels):

    """
    Write individual experiment DataFrames to separate csv files in the specified output folder.
    """
    out_folder = Path(out_folder)
    if not out_folder.exists():
        out_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {out_folder}")

    for df in experiment_dfs:
        quant_cols = [col for col in df.columns if col not in dim_labels]
        if not quant_cols:
            print("Warning: No quantity columns found for individual output.")
            continue

        base_name = quant_cols[0].split("_unc")[0].split("_exp")[0]
        filename = f"{base_name}_GP_results.txt"
        out_path = out_folder / filename
        df.to_csv(out_path, index=False)

        print(f"Written individual output file: {out_path}")

    return

################################################################################

def group_exps_output(experiment_dfs, out_folder, dim_labels):

    """
    Group experiment DataFrames by base quantity name and write merged results to grouped output files.
    """
    out_folder = Path(out_folder)
    if not out_folder.exists():
        out_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {out_folder}")

    grouped_dfs = defaultdict(list)

    for df in experiment_dfs:
        quant_cols = [col for col in df.columns if col not in dim_labels]
        if not quant_cols:
            continue
        base_name = quant_cols[0].split("_unc")[0].split("_exp")[0]
        grouped_dfs[base_name].append(df)

    for base_name, dfs in grouped_dfs.items():
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=dim_labels, how="outer"), dfs)
        filename = f"{base_name}_GP_results.txt"
        out_path = out_folder / filename
        merged_df.to_csv(out_path, index=False)
        print(f"Written combined output file for {base_name}: {out_path}")

    return

################################################################################

def create_GP():

    """
    Main routine to read experiment configuration, perform Gaussian Process fitting,
    and write results to output files.
    Returns a single merged DataFrame of all experiments.
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

        output_folder = out_path
        if not output_folder.exists():
            print(f"Creating output folder: {output_folder}")
            output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {output_folder}")

    else:
        print("Writing combined output file")

        if str(out_path).endswith("/"):
            output_folder = out_path
            if not output_folder.exists():
                print(f"Creating output folder: {output_folder}")
                output_folder.mkdir(parents=True, exist_ok=True)
            combined_output_file = output_folder / "GP_results.txt"
        else:
            combined_output_file = out_path
            output_folder = combined_output_file.parent
            if not output_folder.exists():
                print(f"Creating output folder: {output_folder}")
                output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Output file: {combined_output_file}")

    for file_idx, (file_path, x_known_list, exp_pairs, labels_out) in enumerate(data_list, start=1):
        filename = Path(file_path).stem
        total_experiments = len(x_known_list)
        file_dfs = []

        for idx, (x_known, (y_known, e_known)) in enumerate(zip(x_known_list, exp_pairs)):
            print(f"Processing experiment {idx+1}/{total_experiments} from file {file_idx}/{len(data_list)}: {filename}")

            if x_known.shape[1] == 0:
                print(f"  Skipping experiment {idx+1}: no valid data points.")
                continue

            len_scale = len_scale_opt(x_known, y_known, e_known, MC_progress, MC_plotting, labels_out, out_file_name)
            x_fit = fill_convex_hull(x_known.T, resolution)
            y_fit, e_fit = GP(x_known, y_known, e_known, x_fit.T, len_scale)

            df = pd.DataFrame(x_fit, columns=dim_labels)

            if total_experiments == 1:
                quant_col = f"{filename}"
                err_col = f"{filename}_unc"
            else:
                quant_col = f"{filename}_exp{idx+1}"
                err_col = f"{filename}_unc{idx+1}"

            df[quant_col] = y_fit.flatten()
            df[err_col] = e_fit.flatten()
            file_dfs.append(df)
            experiment_dfs.append(df)

        if write_ind and file_dfs:
            if group_exps:
                merged = reduce(lambda left, right: pd.merge(left, right, on=dim_labels, how="outer"), file_dfs)
                merged = merged.fillna(float('inf'))
                output_path = output_folder / f"{filename}_GP_results.txt"
                merged.to_csv(output_path, index=False)
            else:
                for i, df in enumerate(file_dfs):
                    if total_experiments == 1:
                        output_path = output_folder / f"{filename}_GP_results.txt"
                    else:
                        output_path = output_folder / f"{filename}_exp{i+1}_GP_results.txt"
                    df.to_csv(output_path, index=False)

        print(f"Finished processing {filename}")

    if experiment_dfs:
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=dim_labels, how="outer"), experiment_dfs)
        merged_df = merged_df.fillna(float('inf'))

        if not write_ind:
            merged_df.to_csv(combined_output_file, index=False)
            print(f"Combined results written to {combined_output_file}")

        return merged_df
    else:
        print("No experiment data to return.")
        return pd.DataFrame()

################################################################################

if __name__ == "__main__":
    df = create_GP()
    
