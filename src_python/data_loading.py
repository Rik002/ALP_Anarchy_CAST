import numpy as np
import os
from pathlib import Path
from .constants import CONSTANTS

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
PLOTS_DIR = BASE_DIR / 'plots'

class DataLoading:
    def __init__(self):
        self.DATA_DIR = DATA_DIR
        self.PLOTS_DIR = PLOTS_DIR

    def load_figure_2_data(self, N):
        """
        Load data for Figure 2 from file for a given N.
        Returns: dict with g_e_range, g_gamma_range, W, p_gg_vals, p_eg_vals, phi_osc_vals, u_i0_gamma_vals, u_i0_e_vals
        """
        file_path = self.DATA_DIR / f'figure_2_data_N{N}.txt'
        print(f"DEBUG: Attempting to load: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = np.loadtxt(file_path, skiprows=2)  # Skip header lines
        log_g_e_vals = np.unique(data[:, 0])
        log_g_gamma_vals = np.unique(data[:, 1])
        g_e_count = len(log_g_e_vals)
        g_gamma_count = len(log_g_gamma_vals)

        expected_rows = g_e_count * g_gamma_count
        actual_rows = data.shape[0]
        print(f"DEBUG: Expected rows: {expected_rows} (g_e={g_e_count}, g_gamma={g_gamma_count}), Actual rows: {actual_rows}")

        if actual_rows != expected_rows:
            raise ValueError(f"Cannot reshape data: {actual_rows} rows do not fit grid (g_e={g_e_count}, g_gamma={g_gamma_count})")

        W = data[:, 2].reshape(g_e_count, g_gamma_count)
        p_gg_vals = data[:, 3:3+CONSTANTS['N_REALIZATIONS']].reshape(g_e_count, g_gamma_count, CONSTANTS['N_REALIZATIONS'])
        p_eg_vals = data[:, 3+CONSTANTS['N_REALIZATIONS']:3+2*CONSTANTS['N_REALIZATIONS']].reshape(g_e_count, g_gamma_count, CONSTANTS['N_REALIZATIONS'])
        phi_osc_vals = data[:, 3+2*CONSTANTS['N_REALIZATIONS']:3+3*CONSTANTS['N_REALIZATIONS']].reshape(g_e_count, g_gamma_count, CONSTANTS['N_REALIZATIONS'])
        u_i0_gamma_vals = data[:, 3+3*CONSTANTS['N_REALIZATIONS']:3+3*CONSTANTS['N_REALIZATIONS']+N*CONSTANTS['N_REALIZATIONS']].reshape(g_e_count, g_gamma_count, CONSTANTS['N_REALIZATIONS'], N)
        u_i0_e_vals = data[:, 3+3*CONSTANTS['N_REALIZATIONS']+N*CONSTANTS['N_REALIZATIONS']:].reshape(g_e_count, g_gamma_count, CONSTANTS['N_REALIZATIONS'], N)

        return {
            'g_e_range': log_g_e_vals,
            'g_gamma_range': log_g_gamma_vals,
            'W': W,
            'p_gg_vals': p_gg_vals,
            'p_eg_vals': p_eg_vals,
            'phi_osc_vals': phi_osc_vals,
            'u_i0_gamma_vals': u_i0_gamma_vals,
            'u_i0_e_vals': u_i0_e_vals
        }

    def load_figure_3_data(self):
        """
        Load data for Figure 3.
        Returns: dict with N_range, log_g_50, W_vals, p_gg_vals, u_i0_vals
        """
        file_path = self.DATA_DIR / 'figure_3_data.txt'
        print(f"DEBUG: Attempting to load: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = np.loadtxt(file_path, skiprows=2)
        print(f"DEBUG: Loaded data shape: {data.shape}")

        expected_cols = 1 + CONSTANTS['G_GAMMA_POINTS'] + \
                       CONSTANTS['G_GAMMA_POINTS'] * CONSTANTS['N_REALIZATIONS'] + \
                       CONSTANTS['G_GAMMA_POINTS'] * CONSTANTS['N_REALIZATIONS'] * CONSTANTS['MAX_N']
        if data.shape[1] != expected_cols:
            raise ValueError(f"Expected {expected_cols} columns in {file_path}, but got {data.shape[1]}")

        expected_rows = CONSTANTS['MAX_N'] - 1  # N=2 to MAX_N
        if data.shape[0] != expected_rows:
            print(f"WARNING: Expected {expected_rows} rows (N=2 to MAX_N={CONSTANTS['MAX_N']}), but got {data.shape[0]}")

        N_range = data[:, 0]
        log_g_50_all = data[:, 1:1+CONSTANTS['G_GAMMA_POINTS']]
        p_gg_vals = data[:, 1+CONSTANTS['G_GAMMA_POINTS']:1+CONSTANTS['G_GAMMA_POINTS']+CONSTANTS['G_GAMMA_POINTS']*CONSTANTS['N_REALIZATIONS']].reshape(-1, CONSTANTS['G_GAMMA_POINTS'], CONSTANTS['N_REALIZATIONS'])
        u_i0_vals = data[:, 1+CONSTANTS['G_GAMMA_POINTS']+CONSTANTS['G_GAMMA_POINTS']*CONSTANTS['N_REALIZATIONS']:].reshape(-1, CONSTANTS['G_GAMMA_POINTS'], CONSTANTS['N_REALIZATIONS'], CONSTANTS['MAX_N'])

        print(f"DEBUG: N_range shape: {N_range.shape}, log_g_50_all shape: {log_g_50_all.shape}, p_gg_vals shape: {p_gg_vals.shape}, u_i0_vals shape: {u_i0_vals.shape}")

        W_vals = np.zeros((len(N_range), CONSTANTS['G_GAMMA_POINTS']))
        for i in range(len(N_range)):
            for j in range(CONSTANTS['G_GAMMA_POINTS']):
                g_gamma = 10.0 ** log_g_50_all[i, j]
                g_n1 = np.sqrt(CONSTANTS['G_GAMMA_G_E_LIMIT'] / 1e-15)
                phi_n1 = (CONSTANTS['PHI_P_COEFF'] * g_n1**2 +
                         CONSTANTS['PHI_B_COEFF'] * 1e-15**2 +
                         CONSTANTS['PHI_C_COEFF'] * 1e-15**2)
                phi_max = (g_n1**2 / g_gamma**2) * phi_n1
                W_vals[i, j] = np.sum(p_gg_vals[i, j] * CONSTANTS['PHI_P_COEFF'] * g_gamma**2 < phi_max) / CONSTANTS['N_REALIZATIONS']

        # Select log_g_50 where W is closest to 0.5 for each N
        log_g_50 = np.zeros(len(N_range))
        for i in range(len(N_range)):
            idx_50 = np.argmin(np.abs(W_vals[i] - 0.5))
            log_g_50[i] = log_g_50_all[i, idx_50]

        print(f"DEBUG: log_g_50 shape after selection: {log_g_50.shape}")

        return {
            'N_range': N_range,
            'log_g_50': log_g_50,
            'W_vals': W_vals,
            'p_gg_vals': p_gg_vals,
            'u_i0_vals': u_i0_vals
        }

    def load_convergence_fig2_data(self):
        """
        Load convergence data for Figure 2.
        Returns: N_vals, W_vals
        """
        file_path = self.DATA_DIR / 'convergence_fig2.txt'
        print(f"DEBUG: Attempting to load: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = np.loadtxt(file_path, skiprows=2)
        return data[:, 0], data[:, 1]

    def load_convergence_fig3_data(self):
        """
        Load convergence data for Figure 3.
        Returns: N_vals, W_vals
        """
        file_path = self.DATA_DIR / 'convergence_fig3.txt'
        print(f"DEBUG: Attempting to load: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = np.loadtxt(file_path, skiprows=2)
        return data[:, 0], data[:, 1]

    def load_diagnostics_flux_data(self):
        """
        Load diagnostics flux data.
        Returns: log_g_e, phi_osc, phi_b, phi_c
        """
        file_path = self.DATA_DIR / 'diagnostics_flux.txt'
        print(f"DEBUG: Attempting to load: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = np.loadtxt(file_path, skiprows=2)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    def load_diagnostics_realization_data(self):
        """
        Load diagnostics realization data.
        Returns: N_vals, p_gg_mean, p_gg_sem
        """
        file_path = self.DATA_DIR / 'diagnostics_realization.txt'
        print(f"DEBUG: Attempting to load: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = np.loadtxt(file_path, skiprows=2)
        return data[:, 0], data[:, 1], data[:, 2]

    def load_matrix_distributions(self, N):
        """
        Load matrix distribution data for a given N.
        Returns: U_vals, det_vals
        """
        file_path = self.DATA_DIR / f'matrix_distributions_N{N}.txt'
        print(f"DEBUG: Attempting to load: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = np.loadtxt(file_path, skiprows=2)
        U_vals = data[:, :-1].reshape(-1, N, N)
        det_vals = data[:, -1]
        return U_vals, det_vals
