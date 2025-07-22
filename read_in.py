import numpy as np
import pandas as pd
from pathlib import Path
import yaml

###################################################################################################

def file_has_header(file_path):

    """
    Determines if a csv file has a header row by attempting to parse the first row as numeric.
    """
    
    df_first_row = pd.read_csv(file_path, nrows=1, header=None)
    first_row = df_first_row.iloc[0]

    for val in first_row:
        try:
            float(val)
        except (ValueError, TypeError):
            return True

    return False

#################################################################################################################

def expand_file_paths(file_entries):

    """
    Expands a list of file or folder paths into a list of individual file paths.
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

######################################################################################################################

def header_labels(file_paths, num_kin_dims):

    """
    Returns consistent kinetic dimension labels from file headers, or None if inconsistent or missing.
    """

    header_labels_raw_list = []
    header_labels_norm_list = []

    for file_path in file_paths:
        try:
            if file_has_header(file_path):
                df = pd.read_csv(file_path, nrows=0, header=0)
                header_labels = list(df.columns)
            else:
                continue
        except Exception:
            continue

        if len(header_labels) < num_kin_dims:
            print(f"  File '{file_path}' has insufficient header columns ({len(header_labels)}), skipping.")
            continue

        norm_labels = [label.strip().lower() for label in header_labels[:num_kin_dims]]
        header_labels_raw_list.append(header_labels[:num_kin_dims])
        header_labels_norm_list.append(norm_labels)

    if not header_labels_norm_list:
        return None

    first_norm_labels = header_labels_norm_list[0]
    first_raw_labels = header_labels_raw_list[0]

    for norm_labels in header_labels_norm_list[1:]:
        if norm_labels != first_norm_labels:
            print("Warning: File header labels are inconsistent across files.")
            return None

    return first_raw_labels

##############################################################################################################

def check_data(file_paths, resolution, labels):

    """
    Validates and loads data from the given file paths based on the expected resolution and labels.
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

############################################################################################################################

def read_yaml():

    """
    Parses options.yaml and prepares input settings, file paths, and data for processing.
    """

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

    num_kin_dims = len(resolution)
    data_list = check_data(file_paths, resolution, labels)

    if labels is not None:
        if len(labels) == num_kin_dims:
            kin_labels = labels
            print(f"Using kinetic dimension labels from options.yaml: {kin_labels}")
        else:
            kin_labels = [f"dim{i+1}" for i in range(num_kin_dims)]
            print(f"Warning: The number of kinetic dimension labels ({len(labels)}) does not match the expected number "
                  f"({num_kin_dims}). Using generic kinetic dimension labels instead: {kin_labels}")
    else:
        header_labels = header_labels(file_paths, num_kin_dims)
        if header_labels is not None:
            kin_labels = header_labels
            print(f"Using kinetic dimension labels from file headers: {kin_labels}")
        else:
            kin_labels = [f"dim{i+1}" for i in range(num_kin_dims)]
            print(f"Using generic kinetic dimension labels: {kin_labels}")

    labels = kin_labels + ["quantity", "error"]

    return resolution, MC_progress, MC_plotting, out_file_name, labels, data_list

##########################################################################################################################

def read_data(file_path, labels, resolution):

    """
    Reads and parses data from a file, extracting kinetic dimensions and experimental value/error pairs.
    """

    data = read_csv(file_path)
    num_dims = len(resolution)
    num_cols = data.shape[1]
    num_exp_columns = num_cols - num_dims

    if num_exp_columns % 2 != 0:
        raise ValueError(f"File '{file_path}' must have pairs of columns for quantity and error after kinetic dims.")

    num_experiments = num_exp_columns // 2
    x_known_list = []
    exp_pairs = []
    data_np = data.values

    for exp_idx in range(num_experiments):
        y_col = num_dims + 2 * exp_idx
        e_col = y_col + 1

        y_known = data_np[:, y_col]
        e_known = data_np[:, e_col]
        x_known = data_np[:, :num_dims].T

        valid_mask = np.isfinite(y_known) & np.isfinite(e_known)
        x_known_valid = x_known[:, valid_mask]
        y_known_valid = y_known[valid_mask]
        e_known_valid = e_known[valid_mask]

        x_known_list.append(x_known_valid)
        exp_pairs.append((y_known_valid, e_known_valid))

    return x_known_list, exp_pairs, labels

################################################################################################################################

def read_csv(file_path):

    """
    Reads a csv file using pandas, with or without a header depending on its contents.
    """

    if file_has_header(file_path):
        df = pd.read_csv(file_path, header=0)
    else:
        df = pd.read_csv(file_path, header=None)
    return df
