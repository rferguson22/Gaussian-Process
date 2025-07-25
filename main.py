from GP_fit import create_GP
from calc_prob_surf import generate_prob_surf

###########################################################################################

if __name__ == "__main__":
    df,ndims= create_GP()
    generate_prob_surf(df,ndims)