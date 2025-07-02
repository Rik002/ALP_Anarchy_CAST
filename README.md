# ALP_Anarchy_CAST
Simulation and analysis for the reinterpretation of CAST results. (Independent codebase based on ALP Anarchy by FrancescaChadha-Day et al) 

# ALP Anarchy CAST Simulation Pipeline

This document describes the ALP Anarchy CAST simulation pipeline, including the file structure, purpose of each file, dependencies, installation instructions, how to run the pipeline, and what to edit for customization.

## File Structure

The pipeline is organized in the `alp_anarchy_cast/` directory as follows:

```
alp_anarchy_cast/
├── config/
│   └── constants.dat
├── include_cpp/
│   ├── constants.h
│   ├── flux_calculations.h
│   ├── matrix_operations.h
│   ├── data_output.h
│   └── simulations.h
├── src_cpp/
│   ├── flux_calculations.cpp
│   ├── matrix_operations.cpp
│   ├── data_output.cpp
│   └── simulations.cpp
├── src_python/
│   ├── constants.py
│   ├── data_loading.py
│   ├── statistical_analysis.py
│   └── plotting.py
├── simulations/
│   ├── main.cpp
│   ├── main.py
│   ├── data/ (output directory for C++ data files)
│   └── plots/ (output directory for Python plots)
├── simulate (bash script)
├── plot (bash script)
├── info.md (this file)
```

### File Descriptions

- **config/constants.dat**:
  - **Purpose**: Configuration file defining physical constants, simulation parameters, grid resolution, and output directory.
  - **Format**: INI-style with `key = value : Description, Units`, divided into sections (Physical Constants, Simulation Parameters, Grid Resolution, Directory) separated by `-------`.
  - **Example**:
    ```
    G_GAMMA_N1 = 0.66e-10 : CAST single-ALP bound on photon coupling, GeV^-1
    ```

- **include_cpp/**:
  - **constants.h**: Defines the `Constants` namespace and loads `constants.dat`.
  - **flux_calculations.h**: Declares functions for computing ALP fluxes (Primakoff, Bremsstrahlung, Compton).
  - **matrix_operations.h**: Declares functions for generating random SO(N) matrices and computing probabilities.
  - **data_output.h**: Declares functions for writing simulation data to files.
  - **simulations.h**: Declares functions for generating simulation data (Figures 2, 3, convergence, diagnostics).

- **src_cpp/**:
  - **flux_calculations.cpp**: Implements flux calculations using `long double` for high precision.
  - **matrix_operations.cpp**: Implements random SO(N) matrix generation and probability calculations.
  - **data_output.cpp**: Implements data output functions with high precision (`setprecision(18)`).
  - **simulations.cpp**: Implements simulation logic for Figures 2, 3, convergence, and diagnostics, with OpenMP parallelization and progress monitors.

- **src_python/**:
  - **constants.py**: Loads `constants.dat` for Python scripts using `configparser`.
  - **data_loading.py**: Loads simulation data from `simulations/data/` with error checking.
  - **statistical_analysis.py**: Performs statistical tests (chi-squared, KS tests, curve fitting) with `tqdm` progress bars.
  - **plotting.py**: Generates all plots (heatmaps, histograms, Q-Q plots) with `tqdm` and tabulated summaries.

- **simulations/**:
  - **main.cpp**: Entry point for C++ simulations, runs all tasks (Figures 2, 3, convergence, diagnostics) with timing and tabulated summary.
  - **main.py**: Entry point for Python analysis and plotting, generates all plots with `tqdm` and tabulated summary.
  - **data/**: Directory for simulation output files (e.g., `figure_2_data_N2.txt`).
  - **plots/**: Directory for generated plots (e.g., `figure_2_main_N2.png`).

- **simulate**:
  - **Purpose**: Bash script to compile and run the C++ simulation pipeline.
  - **Features**: Displays constants table, checks for `constants.dat`, compiles with `g++`, runs `simulations/simulate`, and summarizes output files.

- **plot**:
  - **Purpose**: Bash script to run the Python analysis and plotting pipeline.
  - **Features**: Displays constants table, checks for `constants.dat` and data files, installs Python dependencies, runs `simulations/main.py`, and summarizes output files.

- **info.md**:
  - This file, documenting the pipeline.

## Dependencies

### C++ Dependencies
- **g++**: C++ compiler (version 7.0 or later recommended for C++17 and OpenMP support).
- **Eigen**: Linear algebra library for matrix operations.
- **OpenMP**: For parallelization (optional, can be disabled if unavailable).

### Python Dependencies
- **Python 3.8+**
- **numpy**: For numerical computations.
- **scipy**: For statistical analysis.
- **matplotlib**: For plotting.
- **tabulate**: For tabulated output.
- **tqdm**: For progress bars.

## Installation Instructions

### C++ Dependencies
1. **Install g++**:
   - **Ubuntu/Debian**:
     ```bash
     sudo apt update
     sudo apt install g++
     ```
   - **macOS** (via Homebrew):
     ```bash
     brew install gcc
     ```
   - **Windows**: Use MinGW or WSL with the above Ubuntu commands.

2. **Install Eigen**:
   - Download Eigen from [eigen.tuxfamily.org](http://eigen.tuxfamily.org) (version 3.4.0 recommended).
   - Extract to a directory, e.g., `/path/to/eigen`.
   - Update the `simulate` script to point to this directory:
     ```bash
     g++ -Iinclude_cpp -I/path/to/eigen ...
     ```

3. **Install OpenMP** (optional):
   - Included with `g++` on most systems. If unavailable, remove `-fopenmp` from `simulate` and `#pragma omp` directives from `simulations.cpp`.

### Python Dependencies
1. **Install Python 3**:
   - **Ubuntu/Debian**:
     ```bash
     sudo apt update
     sudo apt install python3 python3-pip
     ```
   - **macOS** (via Homebrew):
     ```bash
     brew install python3
     ```
   - **Windows**: Download from [python.org](https://www.python.org).

2. **Install Python Packages**:
   - The `plot` script automatically installs dependencies via `pip`:
     ```bash
     pip install numpy scipy matplotlib tabulate tqdm
     ```
   - Alternatively, install manually:
     ```bash
     pip install numpy scipy matplotlib tabulate tqdm
     ```

## Running the Pipeline

1. **Set Up Directory**:
   - Ensure all files are in the `alp_anarchy_cast/` directory as shown above.
   - Make scripts executable:
     ```bash
     chmod +x simulate plot
     ```

2. **Run Simulations**:
   - From `alp_anarchy_cast/`:
     ```bash
     ./simulate
     ```
   - This:
     - Checks for `config/constants.dat`.
     - Displays the constants table and data resolution.
     - Compiles C++ code with `g++`.
     - Runs `simulations/simulate`, generating data in `simulations/data/`.
     - Prints progress monitors and a tabulated summary of tasks.
     - Lists output files.

3. **Run Analysis and Plotting**:
   - From `alp_anarchy_cast/`:
     ```bash
     ./plot
     ```
   - This:
     - Checks for `config/constants.dat` and `simulations/data/` files.
     - Displays the constants table and data resolution.
     - Installs Python dependencies if missing.
     - Runs `simulations/main.py`, generating plots in `simulations/plots/`.
     - Prints progress bars (`tqdm`) and tabulated summaries for each plotting task.
     - Lists output files.

## Expected Outputs

- **Data Files** (`simulations/data/`):
  - `figure_2_data_N2.txt`, `figure_2_data_N10.txt`, `figure_2_data_N30.txt`: Data for Figure 2 (W, p_gg, p_eg, phi_osc, u_i0).
  - `figure_3_data.txt`: Data for Figure 3 (N, log_g_50, W, p_gg, u_i0).
  - `convergence_fig2.txt`, `convergence_fig3.txt`: Convergence data.
  - `diagnostics_flux.txt`, `diagnostics_realization.txt`, `matrix_distributions_N*.txt`: Diagnostic data.

- **Plots** (`simulations/plots/`):
  - `figure_2_main_N*.png`: W heatmaps for N=2, 10, 30.
  - `figure_2_residuals_N*.png`: Residual heatmaps.
  - `figure_2_chi2_N*.png`: Chi-squared p-value heatmaps.
  - `figure_2_p_gg_qq_N*.png`: Q-Q plots for p_gg.
  - `figure_2_residuals_hist_N*.png`: Residual histograms.
  - `figure_3.png`: log10(g_gamma) vs N.
  - `figure_3_residuals_hist.png`, `figure_3_residuals_qq.png`: Figure 3 residuals.
  - `distributions_N*.png`: p_gg, p_eg, u_i0, matrix element distributions.
  - `matrix_distributions_N*.png`: Matrix elements and determinants.
  - `diagnostic_distributions_fig3.png`: Flux distributions.
  - `convergence_plots.png`: Convergence for Figures 2 and 3.
  - `diagnostics_realization.png`: SEM of p_gg vs N.

## Editing and Customization

- **config/constants.dat**:
  - **What to Edit**: Modify physical constants, simulation parameters, grid resolution, or output directory.
  - **Examples**:
    - Change `N_REALIZATIONS` for more/fewer realizations (affects simulation time and statistical accuracy).
    - Adjust `G_E_LOG_MIN`, `G_E_LOG_MAX`, `G_GAMMA_LOG_MIN`, `G_GAMMA_LOG_MAX` to change the parameter grid.
    - Increase `G_E_POINTS_PER_DECADE`, `G_GAMMA_POINTS_PER_DECADE` for higher resolution (increases computation time).
    - Update `DATA_DIR` to change the output directory (ensure it ends with `/`).

- **simulations/main.cpp**:
  - **What to Edit**: Modify `N_values[]` to change the N values for Figure 2 (e.g., add `15` for N=15).
  - **Example**:
    ```cpp
    int N_values[] = {2, 10, 15, 30};
    ```

- **simulations/main.py**:
  - **What to Edit**: Update `N_values` to match `main.cpp` for plotting Figure 2 data.
  - **Example**:
    ```python
    N_values = [2, 10, 15, 30]
    ```

- **simulate**:
  - **What to Edit**: Update the Eigen include path (`-I/path/to/eigen`) to match your Eigen installation.
  - Remove `-fopenmp` if OpenMP is unavailable.

- **plot**:
  - **What to Edit**: Modify Python dependency installation if using a specific `pip` version (e.g., `pip3`).
  - **Example**:
    ```bash
    pip3 install numpy scipy matplotlib tabulate tqdm
    ```

- **src_cpp/simulations.cpp**:
  - **What to Edit**: Adjust progress update frequency (e.g., change `total_steps / 10` to `total_steps / 20` for more frequent updates).
  - Add custom simulation tasks or modify data output formats.

- **src_python/plotting.py**:
  - **What to Edit**: Customize plot aesthetics (e.g., change `cmap`, `figsize`, or add annotations).
  - Add new statistical tests or plots as needed.

## Notes

- **Precision**: The pipeline uses `long double` in C++ for high precision and `np.float64` in Python. For arbitrary precision, consider MPFR (requires additional setup).
- **Performance**: OpenMP is used for parallelization. For large `N` or `N_REALIZATIONS`, consider GPU parallelization or HDF5 for data storage.
- **Error Handling**: The pipeline includes robust error checking (e.g., file existence, data consistency). Check console output for diagnostics.
- **Extensibility**: Add command-line arguments to `simulate` and `plot` for selective task execution (e.g., `./simulate --N 2`).

For further assistance, contact the pipeline maintainer or refer to the code comments for implementation details.
