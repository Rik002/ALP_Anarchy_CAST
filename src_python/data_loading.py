# Enhanced Data Loading Module
import numpy as np
import os
from pathlib import Path
from constants import CONSTANTS
import warnings

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
PLOTS_DIR = BASE_DIR / 'plots'

class DataLoading:
    def __init__(self):
        self.DATA_DIR = DATA_DIR
        self.PLOTS_DIR = PLOTS_DIR

    def _count_header_lines(self, file_path):
        """Count the number of header lines starting with #"""
        if not file_path.exists():
            return 0
        count = 0
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('#'):
                        count += 1
                    elif line.strip() == '':
                        continue # Skip empty lines
                    else:
                        break # First non-comment, non-empty line
        except:
            pass
        return count

    def _load_data_file(self, filename, expected_cols, subdir=None, auto_skip=True, allow_fewer_cols=False):
        """Enhanced data file loader with better error handling and format validation."""
        if subdir:
            file_path = self.DATA_DIR / subdir / filename
        else:
            file_path = self.DATA_DIR / filename

        print(f"DEBUG: Attempting to load file: {file_path}")
        print(f"DEBUG: File exists: {file_path.exists()}")

        if not file_path.exists():
            if file_path.parent.exists():
                available_files = list(file_path.parent.glob('*'))
                print(f"DEBUG: Available files in {file_path.parent}: {[f.name for f in available_files]}")
            raise ValueError(f"File not found: {file_path}")

        # Read file manually to handle NaN rows and variable columns
        data_rows = []
        skip_header = self._count_header_lines(file_path) if auto_skip else 0

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Skip header lines
        data_lines = lines[skip_header:]

        for line_num, line in enumerate(data_lines, start=skip_header + 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line and check for valid data
            parts = line.split()

            # Handle variable column counts if allowed
            if allow_fewer_cols and len(parts) < expected_cols:
                # Pad with NaN for missing columns
                parts.extend(['nan'] * (expected_cols - len(parts)))
            elif len(parts) != expected_cols and not allow_fewer_cols:
                print(f"DEBUG: Skipping malformed line {line_num}: {line} (Expected {expected_cols} cols, got {len(parts)})")
                continue

            # Check if first element is 'nan' (string) or if any element is invalid
            if parts[0].lower() == 'nan':
                print(f"DEBUG: Skipping NaN placeholder row at line {line_num}: {line}")
                continue

            # Try to convert to float and check for NaN
            try:
                row = []
                for i, part in enumerate(parts):
                    if part.lower() in ['nan', 'inf', '-inf']:
                        if allow_fewer_cols:
                            row.append(np.nan)
                        else:
                            raise ValueError(f"Invalid value: {part}")
                    else:
                        row.append(float(part))

                # Skip rows with any NaN values unless explicitly allowed
                if not allow_fewer_cols and any(np.isnan(x) for x in row if not np.isnan(x)):
                    print(f"DEBUG: Skipping row with NaN values at line {line_num}: {line}")
                    continue

                data_rows.append(row)

            except ValueError as e:
                print(f"DEBUG: Skipping invalid numeric row at line {line_num}: {line} (Error: {e})")
                continue

        if not data_rows:
            raise ValueError(f"No valid data rows in file: {file_path}")

        data = np.array(data_rows)
        print(f"DEBUG: Successfully loaded {data.shape[0]} rows and {data.shape[1]} columns from {filename}")

        if data.shape[1] != expected_cols:
            if allow_fewer_cols:
                print(f"WARNING: Data shape variance in {file_path}: expected {expected_cols} columns, got {data.shape[1]}")
            else:
                raise ValueError(f"Data shape mismatch in {file_path}: expected {expected_cols} columns, got {data.shape[1]}")

        return data

    def load_figure_2_data(self, N):
        """Load data for Figure 2 from file for a given N."""
        # C++ writes: log_g_e log_g_gamma W P_gg_mean P_gg_std P_eg_mean P_eg_std W_residual_mean W_chi2_p_value
        data = self._load_data_file(f'figure_2_N{N}.txt', 9, subdir='data_fig2')

        # Extract unique values for grid - ensure proper sorting
        unique_log_g_e = np.unique(data[:, 0])
        unique_log_g_gamma = np.unique(data[:, 1])
        g_e_count = len(unique_log_g_e)
        g_gamma_count = len(unique_log_g_gamma)

        print(f"DEBUG: Figure 2 grid dimensions: {g_e_count} x {g_gamma_count}")
        print(f"DEBUG: Total data points: {len(data)}, Expected: {g_e_count * g_gamma_count}")

        try:
            return {
                'g_e_range': unique_log_g_e,
                'g_gamma_range': unique_log_g_gamma,
                'W': data[:, 2].reshape(g_e_count, g_gamma_count),
                'P_gg_mean': data[:, 3].reshape(g_e_count, g_gamma_count),
                'P_gg_std': data[:, 4].reshape(g_e_count, g_gamma_count),
                'P_eg_mean': data[:, 5].reshape(g_e_count, g_gamma_count),
                'P_eg_std': data[:, 6].reshape(g_e_count, g_gamma_count),
                'W_residual_mean': data[:, 7].reshape(g_e_count, g_gamma_count),
                'W_chi2_p_value': data[:, 8].reshape(g_e_count, g_gamma_count),
                'raw_data': data  # Keep raw data for additional analysis
            }
        except ValueError as e:
            raise ValueError(f"Error reshaping Figure 2 data for N={N}: {str(e)}")

    def load_figure_3_data(self):
        """Load data for Figure 3."""
        # C++ writes: N log_g_50gamma W P_gg_mean P_gg_std P_eg_mean P_eg_std
        data = self._load_data_file('figure_3.txt', 7, subdir='data_fig3')
        return {
            'N_range': data[:, 0].astype(int),
            'log_g_50gamma': data[:, 1],
            'W': data[:, 2],
            'P_gg_mean': data[:, 3],
            'P_gg_std': data[:, 4],
            'P_eg_mean': data[:, 5],
            'P_eg_std': data[:, 6],
            'raw_data': data
        }

    def load_precomputed_probabilities(self, N, figure_type='fig2'):
        """Load precomputed P_gg and P_eg for histograms."""
        # C++ writes: P_gg P_eg
        if figure_type == 'fig3':
            subdir = 'data_fig3'
        else:
            subdir = 'data_fig2'
        data = self._load_data_file(f'precomputed_probabilities_N{N}.txt', 2, subdir=subdir)
        return data

    def load_cdf_p_gg(self, N, figure_type='fig2'):
        """Compute CDF data for P_gg from precomputed probabilities."""
        probs_data = self.load_precomputed_probabilities(N, figure_type)
        p_gg = np.sort(probs_data[:, 0])
        cdf = np.arange(1, len(p_gg) + 1) / len(p_gg)
        return {'P_gg': p_gg, 'CDF': cdf}

    def load_fit_plot_data(self):
        """Load fit residuals for Figure 3."""
        # C++ writes: N log_g_50gamma predicted_log_g_50gamma residual
        data = self._load_data_file('fit_plot_data.txt', 4, subdir='data_fig3')
        return {
            'N': data[:, 0].astype(int),
            'log_g_50gamma': data[:, 1],
            'predicted': data[:, 2],
            'residual': data[:, 3]
        }

    def load_power_law_fit(self):
        """Load power-law fit parameters."""
        # C++ writes: m c r_squared
        data = self._load_data_file('power_law_fit.txt', 3, subdir='data_fig3')
        return {'m': data[0, 0], 'c': data[0, 1], 'r_squared': data[0, 2]}

    def load_matrix_distributions(self, N):
        """Load matrix data for histograms."""
        # C++ writes: matrix_element det_error orth_error
        data = self._load_data_file(f'matrix_distributions_N{N}.txt', 3, subdir='data_fig2')
        return {
            'matrix_elements': data[:, 0],
            'det_errors': data[:, 1],
            'orth_errors': data[:, 2]
        }

    def load_flux_vs_g_gamma(self):
        """Load flux vs log_g_gamma data."""
        # C++ writes: log_g_gamma phi_p phi_b phi_c
        data = self._load_data_file('flux_vs_g_gamma.txt', 4, subdir='data_flux_comparison')
        return {
            'log_g_gamma': data[:, 0],
            'phi_p': data[:, 1],
            'phi_b': data[:, 2],
            'phi_c': data[:, 3]
        }

    def load_flux_vs_g_e(self):
        """Load flux vs log_g_e data."""
        # C++ writes: log_g_e phi_p phi_b phi_c
        data = self._load_data_file('flux_vs_g_e.txt', 4, subdir='data_flux_comparison')
        return {
            'log_g_e': data[:, 0],
            'phi_p': data[:, 1],
            'phi_b': data[:, 2],
            'phi_c': data[:, 3]
        }

    def load_flux_3d_grid(self):
        """Load 3D flux grid data."""
        # C++ writes: log_g_e log_g_gamma phi_p phi_b phi_c
        data = self._load_data_file('flux_3d_grid.txt', 5, subdir='data_flux_comparison')

        # Extract unique values and reshape
        log_g_e_vals = np.unique(data[:, 0])
        log_g_gamma_vals = np.unique(data[:, 1])
        g_e_count = len(log_g_e_vals)
        g_gamma_count = len(log_g_gamma_vals)

        print(f"DEBUG: 3D Flux grid dimensions: {g_e_count} x {g_gamma_count}")

        try:
            # Force NumPy arrays to ensure .size attribute
            reshaped_phi_p = np.reshape(data[:, 2], (g_e_count, g_gamma_count))
            reshaped_phi_b = np.reshape(data[:, 3], (g_e_count, g_gamma_count))
            reshaped_phi_c = np.reshape(data[:, 4], (g_e_count, g_gamma_count))
            return {
                'log_g_e': log_g_e_vals,
                'log_g_gamma': log_g_gamma_vals,
                'phi_p': reshaped_phi_p,
                'phi_b': reshaped_phi_b,
                'phi_c': reshaped_phi_c
            }
        except ValueError as e:
            raise ValueError(f"Error reshaping 3D flux data: {str(e)}")

    def load_convergence_analysis(self):
        """Load convergence data for all N values."""
        # C++ writes: N n_realizations P_gg_mean P_gg_std P_eg_mean P_eg_std
        data = self._load_data_file('convergence_analysis_all_N.txt', 6, subdir='data_fig2')
        return {
            'N': data[:, 0].astype(int),
            'n_realizations': data[:, 1].astype(int),
            'P_gg_mean': data[:, 2],
            'P_gg_std': data[:, 3],
            'P_eg_mean': data[:, 4],
            'P_eg_std': data[:, 5],
            'raw_data': data
        }

    def load_diagnostics_flux_data(self):
        """Load diagnostics flux data."""
        # C++ writes: phi_osc phi_p phi_b phi_c
        data = self._load_data_file('flux_diagnostics.txt', 4)
        return {
            'phi_osc': data[:, 0],
            'phi_p': data[:, 1],
            'phi_b': data[:, 2],
            'phi_c': data[:, 3]
        }

    def load_diagnostics_realization_data(self):
        """Load diagnostics realization data."""
        # C++ writes: P_gg P_eg
        data = self._load_data_file('realization_diagnostics.txt', 2)
        return {
            'P_gg': data[:, 0],
            'P_eg': data[:, 1]
        }

    def debug_available_files(self):
        """Debug function to list all available files in data directories."""
        print("=== DEBUG: Available Data Files ===")
        base_dirs = ['data_fig2', 'data_fig3', 'data_flux_comparison', '.']
        for subdir in base_dirs:
            dir_path = self.DATA_DIR / subdir if subdir != '.' else self.DATA_DIR
            if dir_path.exists():
                print(f"\n{subdir}:")
                files = list(dir_path.glob('*.txt'))
                for f in sorted(files):
                    file_size = f.stat().st_size if f.exists() else 0
                    print(f"  - {f.name} ({file_size} bytes)")
            else:
                print(f"\n{subdir}: Directory does not exist")

    def validate_data_integrity(self):
        """Validate that all expected data files exist and have reasonable content."""
        validation_results = []

        # Critical files that must exist
        critical_files = [
            ('data_fig2', 'figure_2_N2.txt', 9),
            ('data_fig2', 'figure_2_N10.txt', 9),
            ('data_fig2', 'figure_2_N30.txt', 9),
            ('data_fig3', 'figure_3.txt', 7),
            ('data_fig2', 'precomputed_probabilities_N2.txt', 2),
            ('data_fig2', 'precomputed_probabilities_N10.txt', 2),
            ('data_fig2', 'precomputed_probabilities_N30.txt', 2),
        ]

        for subdir, filename, expected_cols in critical_files:
            try:
                data = self._load_data_file(filename, expected_cols, subdir=subdir)
                status = "✓ PASS" if len(data) > 0 else "⚠ EMPTY"
                validation_results.append((f"{subdir}/{filename}", status, len(data)))
            except Exception as e:
                validation_results.append((f"{subdir}/{filename}", f"✗ FAIL: {str(e)}", 0))

        return validation_results
