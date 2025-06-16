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
- MC_progress: Boolean. Whether to display the MCMC progress. By default is False. 
- out_file_name: String. File path and filename of ouput GP results. By default is in the same folder as the input file and called "GP_results.txt".  
- MC_plotting: Boolean. Whether to plot the MCMC corner plots, KDE peaks and silhouette scores and save these in the same folder as the output GP results (see out_file_name). By default is False. 
- labels: List of Strings. Labels for the column of the output file. By default uses the same labels as the input file if they exist, otherwise uses generic labels dim1, dim2, ..., quantity, error. 

=======================================================
CITATION

If you use this code, please cite:

> R. F. Ferguson, D. G. Ireland & B. McKinnon, “Data‑driven Approach for Interpolation of Sparse Data”, arXiv:2505.01473 (2025).

```bibtex
@misc{ferguson2025data,
  author       = {R. F. Ferguson and D. G. Ireland and B. McKinnon},
  title        = {Data‑driven Approach for Interpolation of Sparse Data},
  year         = {2025},
  eprint       = {2505.01473},
  archivePrefix= {arXiv},
  primaryClass = {physics.data‑an},
  doi          = {10.48550/arXiv.2505.01473}
}
