# Gaussian Process with MCMC Length Scale Optimisation

This Python script generates a **Gaussian Process (GP)** using **Markov Chain Monte Carlo (MCMC)** to optimise the length scale from a novel loss function.  
See the accompanying paper: [*Data‑driven Approach for Interpolation of Sparse Data*](https://arxiv.org/abs/2505.01473).


---

## File Formatting Assumptions

For an **n-dimensional** problem, the input file must contain **n + 2 columns**:

1. First `n` columns: Input kinematic variables  
2. The `(n+1)`th column: Physics quantity of interest  
3. The `(n+2)`th column: Associated error of the physics quantity

The `options.yaml` file must also contain a list named `resolution` with **n** floats or integers for an **n-dimensional** problem.


---

## Optional Parameters in `options.yaml`

| Parameter        | Type            | Description                                                                                         | Default                          |
|------------------|-----------------|-----------------------------------------------------------------------------------------------------|----------------------------------|
| `MC_progress`    | Boolean         | Whether to display the MCMC progress.                                                              | `False`                          |
| `out_file_name`  | String          | File path and filename for output GP results.                                                      | `"GP_results.txt"` (same folder) |
| `MC_plotting`    | Boolean         | Whether to plot and save MCMC corner plots, KDE peaks, and silhouette scores.                      | `False`                          |
| `labels`         | List of Strings | Labels for columns in the output file. Uses input file labels if available, or generic ones (`dim1`, `dim2`, ..., `quantity`, `error`). | *Input-based or generic*         |

---

## Citation

If you use this code, please cite:

> **R.F. Ferguson, D.G. Ireland & B. McKinnon**,  
> *Data‑driven Approach for Interpolation of Sparse Data*,  
> arXiv:2505.01473 (2025).

```bibtex
@misc{GP_ferguson,
  author       = {R.F. Ferguson and D.G. Ireland and B. McKinnon},
  title        = {Data‑driven Approach for Interpolation of Sparse Data},
  year         = {2025},
  eprint       = {2505.01473},
  archivePrefix= {arXiv},
  primaryClass = {physics.data‑an},
  doi          = {10.48550/arXiv.2505.01473}
}
