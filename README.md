Python script to generate a Gaussian Process using a Markov Chain Monte Carlo for length scale optimisation from a novel loss function. 


================================  <br>
File Formatting Assumption:<br>
For an n-dimensional problem, there should be n+2 columns in the input file.
- First n columns are the input kinematic variables
- n+1 column is the physics quantity of interest
- n+2 column is the associated error of the physics quantity

It is also assumed that for an n-dimensional problem, the resolution list in options.yaml should contain n floats/integers. 

================================ <br>
Optional parameters in options.yaml: <br>
- plot: Boolean. Whether to display the MCMC progress and related plotting. By default is False. 
- out_file_name: String. File path and filename of ouput GP results. By default is in the same folder as the input file and called "GP_results.txt".   