import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from functools import reduce
from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP

################################################################################

def create_GP():

    file_paths, resolution, MC_progress, MC_plotting, out_file_name, labels = read_yaml()
    data_list = check_data(file_paths, resolution, labels)

    experiment_dfs = []
    num_dims = len(resolution)
    dim_labels = [f"dim{i+1}" for i in range(num_dims)]

    total_files = len(data_list)

    for file_idx, (file_path, x_known_list, exp_pairs, labels_out) in enumerate(data_list, start=1):
        filename = Path(file_path).stem
        print(f"Processing file {file_idx}/{total_files}: {filename}")

        total_experiments = len(x_known_list)

        for idx, (x_known, (y_known, e_known)) in enumerate(zip(x_known_list, exp_pairs)):
            print(f"  Doing experiment {idx+1}/{total_experiments} fits from file {filename}")

            if x_known.shape[1] == 0:
                print(f"  Skipping experiment {idx+1}: no valid data points.")
                continue

            len_scale = len_scale_opt(x_known, y_known, e_known, MC_progress, MC_plotting, labels_out, out_file_name)
            x_fit = fill_convex_hull(x_known.T, resolution)
            y_fit, e_fit = GP(x_known, y_known, e_known, x_fit.T, len_scale)

            df = pd.DataFrame(x_fit, columns=dim_labels)

            quant_col = f"{filename}_exp{idx+1}"
            err_col = f"{filename}_unc{idx+1}"

            df[quant_col] = y_fit.flatten()
            df[err_col] = e_fit.flatten()

            experiment_dfs.append(df)

        print(f"Finished processing {filename}")

    if not experiment_dfs:
        print("No data processed.")
        return

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=dim_labels, how="outer"), experiment_dfs)
    merged_df = merged_df.fillna(float('inf'))
    merged_df.to_csv(out_file_name, index=False)
    print(f"Combined results written to {out_file_name}")


    return

################################################################################

def check_data(file_paths, resolution, labels):
    """
    Reads and validates all files. Ensures structure and dimensionality match.
    Returns a list of (file_path, x_known_list, exp_pairs, labels).
    """

    data_list = []

    for file_path in file_paths:
        try:
            x_known_list, exp_pairs, labels_out = read_data(file_path, labels, resolution)

            for x_known in x_known_list:
                if len(resolution) != x_known.shape[0]:
                    raise ValueError(
                        f"File '{file_path}' appears to have kinetic dimension {x_known.shape[0]}, "
                        f"but resolution list has {len(resolution)} elements."
                    )

            data_list.append((file_path, x_known_list, exp_pairs, labels_out))

        except Exception as e:
            raise ValueError(f"Failed to load file '{file_path}': {e}")

    print("All datafile paths are readable.")

    return data_list

################################################################################

def expand_file_paths(file_entries):

    """
    Validates and expands a list of file paths and/or directory paths.
    """
    file_paths = []

    for entry in file_entries:
        path_obj = Path(entry)
        if path_obj.is_file():
            file_paths.append(str(path_obj))
        elif path_obj.is_dir():
            files_in_dir = sorted([str(p) for p in path_obj.glob("*") if p.is_file()])
            if not files_in_dir:
                raise ValueError(f"The folder '{entry}' is empty or contains no readable files.")
            file_paths.extend(files_in_dir)
        else:
            raise ValueError(f"'{entry}' is not a valid file or directory.")

    if not file_paths:
        raise ValueError("No valid input files found. Please check 'file_name' entries.")

    return file_paths

################################################################################

def read_yaml():

    with open("options.yaml", "r") as file:
        options = yaml.safe_load(file)

    file_entries = options["file_name"]
    resolution = options["resolution"]
    MC_progress = options.get("MC_progress", False)
    MC_plotting = options.get("MC_plotting", False)
    out_file_name = options.get("out_file_name", None)
    labels = options.get("labels", None)

    if not isinstance(file_entries, list):
        raise ValueError("Expected 'file_name' to be a list of file paths or folder paths.")

    file_paths = expand_file_paths(file_entries)

    if not all(isinstance(item, (float, int)) for item in resolution):
        raise ValueError("All resolution values must be floats or integers.")

    if out_file_name is None:
        out_file_name = "GP_results.txt"

    if labels is None:
        num_kin_dims = len(resolution)
        labels = [f"dim{i+1}" for i in range(num_kin_dims)]
        labels += ["quantity", "error"]

    return file_paths, resolution, MC_progress, MC_plotting, out_file_name, labels


################################################################################

def read_data(file_path, labels, resolution):
    """
    Reads a single data file and returns:
    - x_known_list: list of kinetic dims arrays filtered per experiment
    - exp_pairs: list of (y_known, e_known) arrays filtered per experiment
    - labels_out: labels used in the file

    Assumes first len(resolution) columns are kinetic dims.
    """

    data = read_csv(file_path)

    num_dims = len(resolution) 

    num_cols = data.shape[1]
    num_exp_columns = num_cols - num_dims
    if num_exp_columns % 2 != 0:
        raise ValueError(
            f"Data file '{file_path}' has {num_exp_columns} columns after kinetic dims; "
            "expected an even number (pairs of quantity and error columns)."
        )

    num_experiments = num_exp_columns // 2

    x_all = data.iloc[:, :num_dims].to_numpy()

    x_known_list = []
    exp_pairs = []

    for i in range(num_experiments):
        y_known_full = data.iloc[:, num_dims + 2*i].to_numpy()
        e_known_full = data.iloc[:, num_dims + 2*i + 1].to_numpy()

        valid_mask = np.isfinite(y_known_full)

        x_known = x_all[valid_mask].T
        y_known = y_known_full[valid_mask]
        e_known = e_known_full[valid_mask]

        x_known_list.append(x_known)
        exp_pairs.append((y_known, e_known))

    
    return x_known_list, exp_pairs, labels


################################################################################

def read_csv(file_path):

    """
    Reads a csv or txt file, detecting whether it has a header.
    """
    df_no_header = pd.read_csv(file_path, header=None)
    first_row = df_no_header.iloc[0]

    if all(isinstance(val, str) for val in first_row):
        df = pd.read_csv(file_path)  # Assume header present
    else:
        labels = generate_labels(len(df_no_header.columns), Path(file_path).stem)
        df = pd.read_csv(file_path, names=labels)

    return df

################################################################################

def generate_labels(num_columns, filename="file"):

    """
    Generates default labels: dim1, dim2, ..., quant1, err1, quant2, err2, ...
    """
    num_dims = num_columns - 2
    if num_dims < 1:
        raise ValueError("Insufficient columns to generate labels.")
    
    labels = [f'dim{i+1}' for i in range(num_dims)]
    num_remaining = num_columns - num_dims

    for i in range(num_remaining // 2):
        labels.append(f'{filename}_quant{i+1}')
        labels.append(f'{filename}_err{i+1}')

    return labels

################################################################################

def output_GP(x_fit, y_fit, e_fit, out_file_name, labels):

    """
    Outputs the GP results to CSV.
    """
    data = np.vstack([x_fit.T, y_fit.T, e_fit.T])
    df = pd.DataFrame(data.T, columns=labels)
    df.to_csv(out_file_name, index=False)

################################################################################

create_GP()



