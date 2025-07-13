import numpy as np
from scipy import stats
from .constants import CONSTANTS

class StatisticalAnalysis:
    def compute_residuals(self, W, g_e_range, g_gamma_range):
        """Compute residuals for W relative to expected values."""
        expected_W = np.ones_like(W) * 0.5  # Example: expected W = 0.5
        return W - expected_W

    def compute_chi2_p_values(self, W, N, g_e_range, g_gamma_range):
        """Compute chi-squared p-values for W."""
        chi2_stats = np.zeros_like(W)
        p_vals = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                observed = W[i, j] * CONSTANTS['N_REALIZATIONS']
                expected = 0.5 * CONSTANTS['N_REALIZATIONS']
                chi2_stats[i, j], p_vals[i, j] = stats.chisquare([observed, CONSTANTS['N_REALIZATIONS'] - observed],
                                                                [expected, CONSTANTS['N_REALIZATIONS'] - expected])
        return p_vals

    def fit_figure_3(self, N_range, log_g_50):
        """Fit log(g_gamma) vs N for Figure 3."""
        slope, c, _, _, _ = stats.linregress(N_range, log_g_50)
        fit_log_g = slope * N_range + c
        residuals = log_g_50 - fit_log_g
        ss_tot = np.sum((log_g_50 - np.mean(log_g_50))**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        return slope, c, fit_log_g, residuals, r_squared

    def kstest(self, data, dist, args=()):
        """Perform Kolmogorov-Smirnov test."""
        return stats.ks_2samp(data, stats.norm.rvs(*args, size=len(data))) if dist == 'norm' else \
               stats.ks_2samp(data, stats.uniform.rvs(*args, size=len(data)))
