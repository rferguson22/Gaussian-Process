import pandas as pd
import numpy as np
from scipy.stats import norm

##########################################################################################################

def sum_gaussians(temp_y,temp_gaus):

    '''
    Calculate the average cumulative probability of multiple Gaussian distributions at points temp_y
    '''

    temp_z=np.zeros(len(temp_y))

    dy=abs(max(temp_y)-min(temp_y))/len(temp_y)

    for i in range(int((len(temp_gaus))/2)):
        temp_z+=norm.cdf(temp_y+dy/2,loc=temp_gaus[2*i],scale=temp_gaus[(2*i)+1])
        temp_z-=norm.cdf(temp_y-dy/2,loc=temp_gaus[2*i],scale=temp_gaus[(2*i)+1])
    temp_prob=(temp_z/int(len(temp_gaus)/2))

    return temp_prob

##########################################################################################################

import os
import yaml
import numpy as np
import pandas as pd

def generate_prob_surf(df, ndims, options_path="options.yaml"):
    '''
    Generate a probability surface by evaluating and aggregating Gaussian mixtures
    for each row of data and save to csv. Output location is determined from options.yaml.
    '''

    points = 100
    output_rows = []

    print("Calculating Probability")

    # Default filename
    default_filename = "prob_surf.txt"
    output_file = default_filename

    # Load options.yaml if it exists
    if os.path.exists(options_path):
        with open(options_path, 'r') as f:
            options = yaml.safe_load(f) or {}

        output_path = options.get("out_file_name", "").strip()

        if output_path:
            if os.path.isdir(output_path):
                output_file = os.path.join(output_path, default_filename)
            else:
                output_file = output_path

    for _, row in df.iterrows():
        row = row.to_numpy()
        temp_gaus = row[ndims:][np.isfinite(row[ndims:])]

        if len(temp_gaus) >= 2 and len(temp_gaus) % 2 == 0:
            means = temp_gaus[::2]
            stds = temp_gaus[1::2]

            ymin = min(means - 3 * stds)
            ymax = max(means + 3 * stds)
            temp_y = np.linspace(ymin, ymax, points)
            temp_prob = sum_gaussians(temp_y, temp_gaus)

            for y_val, p_val in zip(temp_y, temp_prob):
                output_rows.append(list(row[:ndims]) + [y_val, p_val])

    output_df = pd.DataFrame(output_rows, columns=df.columns[:ndims].tolist() + ["quantity", "prob"])
    output_df.to_csv(output_file, index=False)

    print(f"Probability results written to {output_file}")

    return

