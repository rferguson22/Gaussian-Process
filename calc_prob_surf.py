import pandas as pd
import numpy as np
from scipy.stats import norm

##########################################################################################################

def sum_gaussians(temp_y,temp_gaus):

    temp_z=np.zeros(len(temp_y))

    dy=abs(max(temp_y)-min(temp_y))/len(temp_y)

    for i in range(int((len(temp_gaus))/2)):
        temp_z+=norm.cdf(temp_y+dy/2,loc=temp_gaus[2*i],scale=temp_gaus[(2*i)+1])
        temp_z-=norm.cdf(temp_y-dy/2,loc=temp_gaus[2*i],scale=temp_gaus[(2*i)+1])
    temp_prob=(temp_z/int(len(temp_gaus)/2))

    return temp_prob

##########################################################################################################

def generate_prob_surf(df,ndims):

    points=100

    output_rows = []

    print("Calculating Probability")

    output_file="prob_surf.csv"

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