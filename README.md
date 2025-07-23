# Gaussian Process with MCMC Length Scale Optimisation

This Python script generates a **Gaussian Process (GP)** using **Markov Chain Monte Carlo (MCMC)** to optimise the length scale from a novel loss function.  
See the accompanying paper: [*Data‑driven Approach for Interpolation of Sparse Data*](https://arxiv.org/abs/2505.01473).

---

## Input Data Assumptions

- The number of kinematic dimensions `n` is **assumed to be the length of the `resolution` list** in `options.yaml`.
- Input can be provided as:
  - One or more individual input files, and/or
  - One or more folder paths containing input files.
- The program collects and processes all input files from the specified files and folders together.
- Each input file must have exactly `n + 2 * e` columns, where:
  - `n` columns are kinematic inputs,
  - `e` is the number of experiments,
  - Each experiment has two columns: quantity and error.
- The code validates that `(total_columns - n)` is even to confirm proper experiment pairing.
- If an input file contains multiple experiments, and a particular experiment does **not** include a measurement for a given kinematic point, use `inf` as a placeholder in both the `quantity` and `error` columns for that experiment.


---

## Optional Parameters in `options.yaml`

| Parameter        | Type            | Description                                                                                                             | Default                          |
|------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| `MC_progress`    | Boolean         | Whether to display the MCMC progress.                                                                                   | `False`                          |
| `out_file_name`  | String          | File path and filename for output GP results.                                                                           | `"GP_results.txt"` (same folder) |
| `MC_plotting`    | Boolean         | Whether to plot and save MCMC corner plots, KDE peaks, and silhouette scores.                                           | `False`                          |
| `labels`         | List of Strings | Labels for the Kinematic Dimension columns in the output file. Uses input file labels if available, or generic labels (`dim1`, `dim2`, etc.) if not. See the **Output Labels** section below for full details. | *Input-based or generic*         |

---

## Output Labels

Labels for columns inside the single combined output file are constructed as follows:

1. The kinematic dimension labels come first:
   - From `labels` in `options.yaml` if provided and length matches kinematic dims.
   - Otherwise, from matching input file headers if available.
   - Otherwise, generic labels: `dim1, dim2, ..., dimN`.

2. Followed by pairs of experiment labels for **each experiment in each input file**, derived from the input filename:  

   - If an input file has only **one experiment**, columns are labelled:  
        filename, filename_unc

   - If an input file has **multiple experiments** (`e > 1`), columns are labelled:  
        filename_exp1, filename_unc1, filename_exp2, filename_unc2, ...

3. Columns from multiple input files are concatenated in the output file in the order the files are provided.

---

## Output File Naming

- There is a **single combined output file** that aggregates all input data and experiment results.
- The filename is determined by the `out_file_name` parameter in `options.yaml`.  
  If not specified, it defaults to `"GP_results.txt"` and is saved in the same folder as the input.
- The file contains all kinematic dimension columns followed by experiment columns from all input files, labelled as described in the **Output Labels** section.
- If a particular experiment does **not** include a measurement at a given kinematic point, the corresponding quantity and error values in the output will be set to `inf` as a placeholder — mirroring the input file format.

---

## Citation

If you use this code, please cite:

> **R.F. Ferguson, D.G. Ireland & B. McKinnon**,  
> *Data‑driven Approach for Interpolation of Sparse Data*,  
> arXiv:2505.01473 (2025).

```bibtex
@misc{GP_sparse,
      title={Data-driven Approach for Interpolation of Sparse Data}, 
      author={R. F. Ferguson and D. G. Ireland and B. McKinnon},
      year={2025},
      eprint={2505.01473},
      archivePrefix={arXiv},
      primaryClass={physics.data-an},
      url={https://arxiv.org/abs/2505.01473}, 
}
