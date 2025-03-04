Python script to generate a Gaussian Process using a Markov Chain Monte Carlo for length scale optimisation from a novel loss function. 


================================  <br>
File Formatting Assumption:<br>
For an n-dimensional problem, there should be n+2 columns in the input file:
- First n columns are the input kinematic variables.
- n+1 column is the physics quantity of interest.
- n+2 column is the associated error of the physics quantity.

It is also assumed that for an n-dimensional problem, the resolution list in options.yaml should contain n floats/integers. 

================================ <br>
Optional parameters in options.yaml: <br>
- MC_progress: Boolean. Whether to display the MCMC progress and related plotting. By default is False. 
- out_file_name: String. File path and filename of ouput GP results. By default is in the same folder as the input file and called "GP_results.txt".  
- MC_plotting: Boolean. Whether to plot the MCMC corner plots, KDE peaks and silhouette scores. By default is False. Uses  out_file_name to save the plots. 
- labels: List of Strings. Labels for the column of the output file. By default uses the same labels as the input file if they exist, otherwise uses generic labels dim0, dim1, ..., quantity, error. 