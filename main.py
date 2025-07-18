import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP

################################################################################

def create_GP():

    file_paths, resolution, MC_progress, MC_plotting, out_file_name, labels = read_yaml()

    data_list = load_and_validate_all_data(file_paths, resolution, labels)

    for file_path, x_known, y_known, e_known, labels_out in data_list:
        print(f"Processing file: {file_path}")

        len_scale = len_scale_opt(x_known, y_known, e_known, MC_progress, MC_plotting, labels_out, out_file_name)

        x_fit = fill_convex_hull(x_known.T, resolution)
        y_fit, e_fit = GP(x_known, y_known, e_known, x_fit.T, len_scale)

        file_suffix = Path(file_path).stem
        output_file = Path(out_file_name)
        if len(file_paths) > 1 or Path(out_file_name).is_dir():
            output_file = output_file.with_name(f"{output_file.stem}_{file_suffix}{output_file.suffix}")

        output_GP(x_fit, y_fit, e_fit, output_file, labels_out)

        print(f"Finished processing {file_path}")

    print("All files processed successfully.")

################################################################################

def load_and_validate_all_data(file_paths, resolution, labels):

    """
    Reads and validates all files. Ensures structure and dimensionality match.
    Returns a list of (file_path, x, y, e, labels).
    """
    
    data_list = []

    for file_path in file_paths:
        try:
            x_known, y_known, e_known, labels_out = read_data(file_path, labels)

            if len(resolution) != len(x_known):
                raise ValueError(
                    f"File '{file_path}' appears to have {len(x_known)} dimensions, "
                    f"but resolution list has {len(resolution)} elements."
                )

            data_list.append((file_path, x_known, y_known, e_known, labels_out))

        except Exception as e:
            raise ValueError(f"Failed to load file '{file_path}': {e}")

    return data_list

################################################################################

def validate_and_expand_file_paths(file_entries):

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

    file_paths = validate_and_expand_file_paths(file_entries)

    if not all(isinstance(item, (float, int)) for item in resolution):
        raise ValueError("All resolution values must be floats or integers.")

    if out_file_name is None:
        out_file_name = "GP_results.txt"

    return file_paths, resolution, MC_progress, MC_plotting, out_file_name, labels

################################################################################

def read_data(file_path, labels):
    """
    Reads a single data file and returns x, y, error arrays with validated labels.
    """
    data = read_csv(file_path)

    if labels is None:
        labels = data.columns

    if len(labels) != len(data.columns):
        raise ValueError(
            f"Expected {len(data.columns)} column names in labels but received {len(labels)}"
        )

    if len(data.columns) < 3:
        raise ValueError(f"File '{file_path}' must have at least 3 columns (features + quantity + error)")

    x_known = data.iloc[:, :-2].to_numpy().T
    y_known = data.iloc[:, -2].to_numpy().T
    e_known = data.iloc[:, -1].to_numpy().T

    return x_known, y_known, e_known, labels

################################################################################

def read_csv(file_path):
    """
    Reads a CSV or TXT file, detecting whether it has a header.
    """
    df_no_header = pd.read_csv(file_path, header=None)
    first_row = df_no_header.iloc[0]

    if all(isinstance(val, str) for val in first_row):
        df = pd.read_csv(file_path)  # Assume header present
    else:
        labels = generate_labels(len(df_no_header.columns))
        df = pd.read_csv(file_path, names=labels)

    return df

################################################################################

def generate_labels(num_columns):
    """
    Generates default labels for columns: dim1, dim2, ..., quantity, error
    """
    labels = [f'dim{i+1}' for i in range(num_columns - 2)]
    labels.extend(['quantity', 'error'])
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


