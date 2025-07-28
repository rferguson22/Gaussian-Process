
# Gaussian Process with MCMC Length Scale Optimisation

This Python script generates a **Gaussian Process (GP)** using **Markov Chain Monte Carlo (MCMC)** to optimise the length scale from a novel loss function.  
See the accompanying paper: [*Data‑driven Approach for Interpolation of Sparse Data*](https://arxiv.org/abs/2505.01473).

---
## Input Data Assumptions

**File format expectations depend on the `gp_fit` flag in `options.yaml`:**
- If `gp_fit: true`: input files must contain **raw experimental data** to be used for GP fitting.
- If `gp_fit: false`: input files must contain **precomputed GP results** to be used for probability calculation.

Regardless of this setting, the following format rules apply to **all files**.

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

| Parameter                  | Type    | Description                                                                                                                                              | Default             |
|---------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `gp_fit`                  | Boolean | If `true`, run Gaussian Process fitting. If `false`, assume input files are precomputed GP results.                                                     | `True`              |
| `gen_prob_surf`           | Boolean | If `true`, compute and output a probability surface from merged GP results.                                                                              | `False`             |
| `MC_progress`             | Boolean | Whether to display the MCMC progress.                                                                                                                    | `False`             |
| `MC_plotting`             | Boolean | Whether to plot and save MCMC corner plots, KDE peaks, and silhouette scores.                                                                            | `False`             |
| `out_file_name`           | String  | For combined mode: output file path or folder. For individual mode: treated as a folder path.                                                            | `"GP_results.txt"`  |
| `labels`                  | List    | Labels for kinematic dimension columns. See **Output Labels** section.                                                                                   | Auto or generic     |
| `write_individual_files`  | Boolean | If `true`, writes one output file per input file (or per experiment if not grouped). Otherwise, writes a single combined output.                        | `False`             |
| `group_experiments_per_file` | Boolean | If `true`, and using individual output mode, all experiments from the same input file go into one file. Otherwise, each experiment gets its own file. | `False`             |

---

## Output Modes & File Naming

### Combined Output Mode (`write_individual_files: false`)

- A single output file is written.
- If `out_file_name` is a **file path** (i.e. does **not** end with a `/`), output is written directly to that file.
- If `out_file_name` **ends with a `/`**, it is treated as a folder path, and the output is saved inside that folder as `GP_results.txt`.


### Individual Output Mode (`write_individual_files: true`)
- The value of `out_file_name` is treated as a **folder**, whether or not it ends in `/`.
- That folder will be created if it doesn’t exist.
- Each input file produces its own output file(s):
  - If `group_experiments_per_file: true`, one output per file: `filename_GP_results.txt`
  - If `false`, one per experiment: `filename_exp1_GP_results.txt`, etc.

In all cases, missing measurements in any experiment are filled with `inf`.

---

## Output Labels

Labels for columns inside the output file(s) are constructed as follows:

1. The kinematic dimension labels come first:
   - From `labels` in `options.yaml` if provided and length matches kinematic dims.
   - Otherwise, from matching input file headers if available.
   - Otherwise, generic labels: `dim1, dim2, ..., dimN`.

2. Followed by pairs of experiment labels for **each experiment in each input file**, derived from the input filename:  
   - If an input file has only **one experiment**, columns are labelled:  
        `filename`, `filename_unc`
   - If an input file has **multiple experiments**, columns are labelled:  
        `filename_exp1`, `filename_unc1`, `filename_exp2`, `filename_unc2`, ...

3. Columns from multiple input files are merged in the order the files are provided.

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
```
