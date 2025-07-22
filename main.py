import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce

from find_len_scales import len_scale_opt
from convex_hull import fill_convex_hull
from GP_func import GP
from read_in import read_yaml

################################################################################

def create_GP():
    
    resolution, MC_progress, MC_plotting, out_file_name, labels, data_list = read_yaml()

    experiment_dfs = []
    num_dims = len(resolution)
    dim_labels = labels[:num_dims]

    total_files = len(data_list)

    for file_idx, (file_path, x_known_list, exp_pairs, labels_out) in enumerate(data_list, start=1):
        filename = Path(file_path).stem
        total_experiments = len(x_known_list)

        for idx, (x_known, (y_known, e_known)) in enumerate(zip(x_known_list, exp_pairs)):
            print(f"Processing experiment {idx+1}/{total_experiments} from file {file_idx}/{total_files}: {filename}")

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

if __name__ == "__main__":
    create_GP()
